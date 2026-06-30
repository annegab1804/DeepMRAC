import os
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import dicom2nifti
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ants
import xml.etree.ElementTree as ET
from nilearn.maskers import NiftiLabelsMasker


def create_whole_brain_mask(t1_path: str, reg: dict) -> np.ndarray:
    """Performs tissue segmentation and creates a Whole Brain mask in MNI space.
    
    This function applies N3 bias correction to the T1 image, segments it into 
    three tissue classes (CSF, GM, WM) using the Atropos algorithm, and warps 
    the resulting Grey Matter and White Matter probability maps to MNI space 
    to create a binary Whole Brain mask.

    Args:
        t1_path: Path to the input T1-weighted NIfTI file.
        reg: Dictionary containing registration transforms (from ants.registration) 
             between the native T1 and the MNI template.

    Returns:
        A NumPy boolean array representing the Whole Brain mask (GM + WM) 
        in MNI space, thresholded at 0.5.
    """
    # Load images and template
    mni_template = ants.image_read(ants.get_ants_data('mni'))
    t1 = ants.image_read(t1_path)

    # Pre-processing: Generate a brain mask and correct intensity bias
    # The mask limits Atropos to brain tissue, and N3 improves class separation
    t1_mask = ants.get_mask(t1)
    t1_n3 = ants.n3_bias_field_correction(t1)

    # Atropos: 3-class segmentation (1=CSF, 2=GM, 3=WM)
    # We use k-means initialization for the three tissue types
    seg = ants.atropos(
        a=t1_n3,
        m='[0.2,1x1x1]',
        c='[3,0]',
        i='kmeans[3]',
        x=t1_mask
    )

    # Warp masks to MNI space
    # Index 1 is Grey Matter, Index 2 is White Matter
    gm_mni = ants.apply_transforms(fixed=mni_template, moving=seg['probabilityimages'][1],
                                    transformlist=reg['fwdtransforms'])
    wm_mni = ants.apply_transforms(fixed=mni_template, moving=seg['probabilityimages'][2],
                                    transformlist=reg['fwdtransforms'])

    # Create Whole Brain mask in MNI space (Threshold 0.5)
    wb_mask = (gm_mni + wm_mni).numpy() > 0.5

    return wb_mask

def align_pet_to_t1(pet_path: str, t1_path: str, output_path: str) -> None:
    """Performs rigid-body registration to align a PET image to T1 structural space.
    
    Uses a 6-degree-of-freedom (6-DOF) transform to account for subject movement
    between or during scans, ensuring PET voxels overlap correctly with T1 anatomy.

    Args:
        pet_path: Path to the moving PET NIfTI image.
        t1_path: Path to the fixed T1 NIfTI image (the target space).
        output_path: Directory path where the registered PET image will be saved.

    Returns:
        None. Saves 'pet_in_t1.nii.gz' to the output directory.
    """
    # Load images
    pet = ants.image_read(pet_path)
    t1 = ants.image_read(t1_path)

    # Register PET to T1 (Rigid body: 6 degrees of freedom)
    reg = ants.registration(fixed=t1, moving=pet, type_of_transform='Rigid')

    # The registered PET in T1 space
    pet_in_t1 = reg['warpedmovout']
    ants.image_write(pet_in_t1, output_path)

def extracting_roi(
    t1_path: str,
    pet_in_t1_path: str,
    aal_atlas_path: str,
    labels_path: str, 
    target_names: set,
) -> dict:
    """Warps the AAL atlas to native T1 space and extracts mean ROI signals.
    
    This function calculates a non-linear (SyN) registration between the subject 
    T1 and the MNI template, applies the inverse transform to the AAL atlas, 
    and uses the resulting native-space atlas to extract PET values for specific 
    regions of interest.

    Args:
        t1_path: Path to the subject's T1-weighted image.
        pet_in_t1_path: Path to the PET image already coregistered to the T1.
        aal_atlas_path: Path to the AAL atlas NIfTI file (in MNI space).
        labels_path: Path to the AAL atlas XML file containing label indices and names.
        target_names: A set of strings containing the ROI names to extract. 

    Returns:
        A dictionary where keys are ROI names (str) and values are the 
        corresponding mean PET signals (float).
    """
    # Load the reference MNI template and the subject's T1 image
    mni_template = ants.image_read(ants.get_ants_data('mni'))
    t1 = ants.image_read(t1_path)

    # Compute non-linear registration (SyN) from T1 to MNI space
    # This generates the mapping needed to bring data back and forth between spaces
    t1_to_mni = ants.registration(fixed=mni_template, moving=t1, type_of_transform='SyN')

    # Load the AAL atlas and warp it back to the subject's native T1 space
    # We use the inverse transforms from the T1-to-MNI registration
    aal_mni = ants.image_read(aal_atlas_path)
    aal_in_t1 = ants.apply_transforms(
        fixed=t1,
        moving=aal_mni,
        transformlist=t1_to_mni['invtransforms'],
        interpolator='genericLabel'  # CRITICAL: Preserves integer labels
    )

    # Convert the warped ANTs image to a Nibabel object for Nilearn compatibility
    aal_in_t1_nib = ants.to_nibabel_nifti(aal_in_t1)

    # Initialize the NiftiLabelsMasker to extract mean signals
    # We use 'resampling_target="data"' to ensure the atlas matches the PET image dimensions
    masker = NiftiLabelsMasker(
        labels_img=aal_in_t1_nib, 
        standardize=False, 
        resampling_target="data"
    )
    # Extract mean values for all regions present in the warped atlas
    # fit_transform returns a 2D array, we flatten it to get a 1D vector of means
    roi_values = masker.fit_transform(pet_in_t1_path).flatten()

    # Identify which label IDs were actually processed by the masker
    extracted_ids = masker.labels_

    # Parse the AAL XML file to create a mapping between indices and anatomical names
    tree = ET.parse(labels_path)
    root = tree.getroot()
    
    id_to_name = {}
    for label in root.iter('label'):
        idx = label.find('index').text
        name = label.find('name').text
        if name in target_names:
            id_to_name[int(idx)] = name

    # Map the extracted numerical values back to their human-readable names
    results = {}
    for i, label_id in enumerate(extracted_ids):
        # Check if the current ID from the masker is one of our target ROIs
        if label_id in id_to_name:
            region_name = id_to_name[label_id]
            results[region_name] = roi_values[i]

    return results

def uhr_original_dicom_to_nifti_beqml(
    dicom_dir: str, 
    output_nii_path: str,
    factor: float = 79.22,
) -> None:
    """Converts UHR DICOM images to NIfTI format and scales to Bq/mL.

    This function performs the conversion of a DICOM directory into a single 
    NIfTI volume, applies a calibration factor to convert raw counts into 
    physical activity concentration (Bq/mL), and saves the result.

    Args:
        dicom_dir: Path to the directory containing the UHR DICOM slices.
        output_nii_path: Full path (including filename) where the calibrated 
            NIfTI image will be saved.
        factor (float): Calibration factor to convert the voxels in beq/ml.
            Default to 79.22.

    Returns:
        None. The function writes the converted image to disk at output_nii_path.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dicom2nifti.convert_directory(dicom_dir, tmp_dir, compression=True, reorient=True)
        # Find the created file
        converted_file = [f for f in os.listdir(tmp_dir) if f.endswith('.nii.gz')][0]
        uhr_nifti = nib.load(os.path.join(tmp_dir, converted_file))
        activity_data = (uhr_nifti.get_fdata() * factor)
        affine = uhr_nifti.affine.copy() # Use .copy() to be safe

    print(f"Activity data in kBeq/ml- Max: {activity_data.max()}, Mean: {activity_data.mean()}")

    # Save the final result
    final_img = nib.Nifti1Image(activity_data, affine)
    nib.save(final_img, output_nii_path)

def hrrt_original_ecat_to_nifti_beqml(
    file_path: str,
    factor: float,
    output_nii_path: str,
) -> None:
    """Processes an HRRT ECAT7 file, converts to RAS orientation, and saves as NIfTI.

    Args:
        file_path (str): Path to the input ECAT7 (.v) file.
        factor (float): Calibration factor to convert the voxels in beq/ml.
        output_nii_path (str): Path where the resulting relative SUV NIfTI will be saved.

    Returns:
        None
    """
    ecat = nib.ecat.load(file_path)
    data = ecat.get_fdata() 
    affine = ecat.affine

    if data.ndim == 4:
        # Extract first frame
        data_lpi = data[:, :, :, 0]
    else:
        data_lpi = data
    # We assume LPI orientation for HRRT by default
    data_ras = data_lpi[::-1, ::-1, ::-1] * factor

    print(f"Activity data - Max: {data_ras.max()}, Mean: {data_ras.mean()}")

    # Save the final result
    final_img = nib.Nifti1Image(data_ras, affine)
    nib.save(final_img, output_nii_path)

def hrrt_ecat_to_nifti_suv(
    file_path: str,
    factor: float,
    weight: float,
    net_dose: int,
    half_life: int,
    start_time: str,
    injection_time: str,
    output_nii_path: str,
) -> np.ndarray:
    """Processes an HRRT ECAT7 file, converts to RAS orientation, and saves as NIfTI.

    Args:
        file_path (str): Path to the input ECAT7 (.v) file.
        factor (float): Calibration factor to convert the voxels in beq/ml.
        weight (float): The weight of the patient in kg.
        net_dose (int): The injected dose in kBq.
        half_life (int): Half life time of the radioactive tracer injected in seconds.
        start_time (str): The start time of the HRRT scan for decay offset calculation.
            'HH:MM' or 'HH:MM:SS'
        injection_time (str): The time of the injection on the radiative tracer.
            'HH:MM' or 'HH:MM:SS'
        output_nii_path (str): Path where the resulting relative SUV NIfTI will be saved.

    Returns:
        np.ndarray: The normalized relative SUV image data.
    """
    ecat = nib.ecat.load(file_path)
    data = ecat.get_fdata() 
    affine = ecat.affine

    if data.ndim == 4:
        # Extract first frame
        data_lpi = data[:, :, :, 0]
    else:
        data_lpi = data
    # We assume LPI orientation for HRRT by default
    data_ras = data_lpi[::-1, ::-1, ::-1]*factor/1000

    print(f"Activity data in kBeq/ml - Max: {data_ras.max()}, Mean: {data_ras.mean()}")

    delta_t = time_to_seconds(start_time) - time_to_seconds(injection_time)
    if delta_t < 0:
        print(f"Warning: delta_t is negative ({delta_t}s). Check timestamps.")

    # Decay Correction 
    decay_constant = np.log(2) / half_life
    decay_corrected_dose = net_dose * np.exp(-decay_constant * delta_t)
    
    # Normalize to Relative SUV
    hrrt_suv = data_ras * weight / decay_corrected_dose
    print(f"SUV - Max: {hrrt_suv.max()}, Mean: {hrrt_suv.mean()}")

    # Save the final result
    final_img = nib.Nifti1Image(hrrt_suv, affine)
    nib.save(final_img, output_nii_path)
    return hrrt_suv

def uhr_dicom_to_nifti_suv(
    dicom_dir: str, 
    weight: float,
    net_dose: int,
    half_life: int,
    start_time: str,
    injection_time: str,
    output_nii_path: str
) -> np.ndarray:
    """Converts UHR DICOMs to NIfTI, applies decay correction, and normalizes.

    Args:
        dicom_dir (str): Directory containing the UHR DICOM slices.
        weight (float): The weight of the patient in kg.
        net_dose (int): The injected dose in kBq.
        half_life (int): Half life time of the radioactive tracer injected in seconds.
        start_time (str): The start time of the UHR scan for decay offset calculation.
            'HH:MM' or 'HH:MM:SS'
        injection_time (str): The time of the injection on the radiative tracer.
            'HH:MM' or 'HH:MM:SS'
        output_nii_path (str): Path where the resulting relative SUV NIfTI will be saved.

    Returns:
        np.ndarray: The normalized relative SUV image data.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dicom2nifti.convert_directory(dicom_dir, tmp_dir, compression=True, reorient=True)
        # Find the created file
        converted_file = [f for f in os.listdir(tmp_dir) if f.endswith('.nii.gz')][0]
        uhr_nifti = nib.load(os.path.join(tmp_dir, converted_file))
        activity_data = uhr_nifti.get_fdata()
        print(f"Original activity data - Max: {activity_data.max()}, Mean: {activity_data.mean()}")
        activity_data = (uhr_nifti.get_fdata() * 79.22) / 1000.0
        affine = uhr_nifti.affine.copy() # Use .copy() to be safe

    print(f"Activity data - Max: {activity_data.max()}, Mean: {activity_data.mean()}")

    delta_t = time_to_seconds(start_time) - time_to_seconds(injection_time)
    if delta_t < 0:
        print(f"Warning: delta_t is negative ({delta_t}s). Check timestamps.")

    # Decay Correction
    decay_constant = np.log(2) / half_life
    decay_corrected_dose = net_dose * np.exp(-decay_constant * delta_t)
    
    # Normalize to Relative SUV
    uhr_suv = activity_data * weight / decay_corrected_dose
    print(f"SUV - Max: {uhr_suv.max()}, Mean: {uhr_suv.mean()}")

    # Save the final result
    final_img = nib.Nifti1Image(uhr_suv, affine)
    nib.save(final_img, output_nii_path)

    return uhr_suv

def generate_summary_violin(
    cohort_results: dict[str, dict[str, list[float]]], 
    save_path: str
) -> None :
    """Creates a single figure containing violin plots for all ROIs.
    
    Args:
        cohort_results: A dictionary mapping ROI names to UHR and HRRT value lists.
        output_folder: System path where the PNG files will be saved.
    """
    rois = [name for name, data in cohort_results.items() if len(data['UHR']) > 0]
    n_rois = len(rois)
    
    # Calculate grid size (e.g., for 15 ROIs, 5 rows x 3 cols)
    cols = 3
    rows = (n_rois + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    for i, roi_name in enumerate(rois):
        uhr = cohort_results[roi_name]['UHR']
        hrrt = cohort_results[roi_name]['HRRT']
        
        df = pd.DataFrame({
            'SUV': uhr + hrrt,
            'Method': ['UHR'] * len(uhr) + ['HRRT'] * len(hrrt)
        })
        
        sns.violinplot(data=df, x='Method', y='SUV', hue='Method', ax=axes[i],
                        palette={'UHR': 'skyblue', 'HRRT': 'salmon'}, inner="box", legend=False)
        sns.stripplot(data=df, x='Method', y='SUV', color='black', alpha=0.3, ax=axes[i])
        
        axes[i].set_title(f"ROI: {roi_name}")
        axes[i].set_ylabel("Mean SUV")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Summary of SUV Distributions across all ROIs", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    print(f"--- Summary Violin Plot saved: {save_path}")
    plt.close()

def generate_summary_bland_altman(
    cohort_results: dict[str, dict[str, list[float]]], 
    save_path: str
) -> None :
    """Creates a single figure containing Bland-Altman plots for all ROIs.
    
    Args:
        cohort_results: A dictionary mapping ROI names to UHR and HRRT value lists.
        output_folder: System path where the PNG files will be saved.
    """
    rois = [name for name, data in cohort_results.items() if len(data['UHR']) > 1]
    n_rois = len(rois)
    
    cols = 3
    rows = (n_rois + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, roi_name in enumerate(rois):
        d1 = np.array(cohort_results[roi_name]['UHR'])
        d2 = np.array(cohort_results[roi_name]['HRRT'])
        
        mean_val = (d1 + d2) / 2
        diff = d1 - d2
        md = np.mean(diff)
        sd = np.std(diff)

        axes[i].scatter(mean_val, diff, alpha=0.6, edgecolors='k', color='tab:red')
        axes[i].axhline(md, color='black', linestyle='-', label=f'Bias: {md:.2e}')
        axes[i].axhline(md + 1.96*sd, color='gray', linestyle='--')
        axes[i].axhline(md - 1.96*sd, color='gray', linestyle='--')
        
        axes[i].set_title(f"Bland-Altman: {roi_name}")
        axes[i].set_xlabel('Mean SUV')
        axes[i].set_ylabel('Diff (UHR - HRRT)')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Summary of Agreement (Bland-Altman) across all ROIs", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    print(f"--- Summary Bland-Altman saved: {save_path}")
    plt.close()

def generate_summary_percentage_difference(
    cohort_results: dict[str, dict[str, list[float]]], 
    save_path: str
) -> None :
    """Creates a bar plot showing the mean % difference in SUV per ROI.
    
    Args:
        cohort_results: A dictionary mapping ROI names to UHR and HRRT value lists.
        output_folder: System path where the PNG files will be saved.
    """
    roi_names = []
    percent_diffs = []
    stds = []

    for roi_name, data in cohort_results.items():
        uhr = np.array(data['UHR'])
        hrrt = np.array(data['HRRT'])
        
        # Filter out any lingering NaNs
        mask = ~np.isnan(uhr) & ~np.isnan(hrrt)
        uhr, hrrt = uhr[mask], hrrt[mask]

        if len(uhr) > 0:
            # Calculate % difference for each patient in this ROI
            # Avoid division by zero if hrrt is 0
            diff = ((uhr - hrrt) / np.where(hrrt == 0, np.nan, hrrt)) * 100
            
            roi_names.append(roi_name)
            percent_diffs.append(np.nanmean(diff))
            stds.append(np.nanstd(diff))

    if not roi_names:
        print("!!! No data available for Percentage Difference plot.")
        return

    # Create the plot
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # Use a barplot with error bars (standard deviation)
    x_pos = np.arange(len(roi_names))
    bars = plt.bar(x_pos, percent_diffs, yerr=stds, align='center', 
                   alpha=0.7, color='teal', capsize=10, edgecolor='black')

    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-')

    plt.xticks(x_pos, roi_names, rotation=45, ha='right')
    plt.ylabel('SUV % Difference (UHR vs HRRT)')
    plt.title('Mean SUV % Difference per Region of Interest\n(Error bars = ±1 SD)')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (2 if yval > 0 else -5), 
                 f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"--- Percentage Difference Bar Plot saved: {save_path}")
    plt.close()

def time_to_seconds(time_str: str) -> float:
    """Converts format 'HH:MM' or 'HH:MM:SS' to seconds."""
    formats = ["%H:%M:%S", "%H:%M"]
    
    for fmt in formats:
        try:
            t = datetime.strptime(time_str, fmt)
            return t.hour * 3600 + t.minute * 60 + t.second
        except ValueError:
            continue
    raise ValueError(f"Invalid time format : {time_str}")