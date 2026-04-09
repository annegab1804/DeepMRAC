import os
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import dicom2nifti
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time


def hrrt_ecat_to_nifti_suv(file_path: str, output_nii_path: str) -> tuple[np.ndarray, float, time]:
    """Processes an HRRT ECAT7 file, converts to RAS orientation, and saves as NIfTI.

    Args:
        file_path: Path to the input ECAT7 (.v) file.
        output_nii_path: Path where the converted NIfTI file will be saved.

    Returns:
        A tuple containing:
            - data_ras (np.ndarray): The 3D image data in RAS orientation.
            - dosage (float): The injected dose value from the ECAT header.
            - hrrt_time_only (datetime.time): The scan start time as a time object.
    """
    ecat = nib.ecat.load(file_path)
    # ECAT data usually needs to be multiplied by the scale factor in the subheader
    data = ecat.get_fdata() 
    affine = ecat.affine

    # Handle 4D -> 3D
    if data.ndim == 4:
        # Extract first frame
        data_lpi = data[:, :, :, 0]
        # We assume LPI orientation for HRRT by default
        data_ras = data_lpi[::-1, ::-1, ::-1]
    
    # Extract headers (dosage and scan_start_time)
    dosage = ecat.header['dosage']

    # Convert Unix Timestamp to a Python 'time' object (HH:MM:SS)
    unix_time = ecat.header['scan_start_time']
    dt_object = datetime.fromtimestamp(unix_time)
    # If the scanner was in Eastern Daylight Time (UTC-4)
    # offset_seconds = -4 * 3600 
    # dt_object = datetime.fromtimestamp(unix_time + offset_seconds)
    hrrt_time_only = dt_object.time()

    hrrt_suv = data_ras / dosage

    nifti_img = nib.Nifti1Image(hrrt_suv, affine)
    nib.save(nifti_img, output_nii_path)
    
    return hrrt_suv, dosage, hrrt_time_only

def uhr_dicom_to_nifti_suv(
    dicom_dir: str, 
    hrrt_dose: float, 
    hrrt_time: time, 
    output_nii_path: str
) -> np.ndarray:
    """Converts UHR DICOMs to NIfTI, applies decay correction relative to HRRT, and normalizes.

    Args:
        dicom_dir: Directory containing the UHR DICOM slices.
        hrrt_dose: The dose value from the HRRT scan to be used as baseline.
        hrrt_time: The start time of the HRRT scan for decay offset calculation.
        output_nii_path: Path where the resulting relative SUV NIfTI will be saved.

    Returns:
        np.ndarray: The normalized relative SUV image data.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dicom2nifti.convert_directory(dicom_dir, tmp_dir, compression=True, reorient=True)
        # Find the created file
        converted_file = [f for f in os.listdir(tmp_dir) if f.endswith('.nii.gz')][0]
        uhr_nifti = nib.load(os.path.join(tmp_dir, converted_file))

        # We call .get_fdata() HERE while the file still exists
        activity_data = uhr_nifti.get_fdata() * 66
        affine = uhr_nifti.affine.copy() # Use .copy() to be safe

    #  Extract Metadata for Decay Correction from the first DICOM slice
    sample_slice = pydicom.dcmread(os.path.join(dicom_dir, os.listdir(dicom_dir)[0]))
    
    # Convert DICOM HHMMSS to a time object
    uhr_time_str = sample_slice.SeriesTime.split('.')[0]
    uhr_time = datetime.strptime(uhr_time_str, '%H%M%S').time()

    # Calculate delta_t relative to HRRT scan start
    dummy_date = datetime(2000, 1, 1)
    delta_t = (datetime.combine(dummy_date, uhr_time) - 
               datetime.combine(dummy_date, hrrt_time)).total_seconds()

    # Decay Correction using HRRT Dosage as baseline
    half_life = float(sample_slice.get("RadionuclideHalfLife", 6588))
    decay_constant = np.log(2) / half_life
    decay_corrected_dose = hrrt_dose * np.exp(-decay_constant * delta_t)
    
    # Normalize to Relative SUV
    # Note: dicom2nifti usually applies the RescaleSlope during conversion.
    # If the voxel values are already calibrated, we just divide by dose.
    uhr_suv = activity_data / decay_corrected_dose

    # Save the final result
    final_img = nib.Nifti1Image(uhr_suv, affine)
    nib.save(final_img, output_nii_path)

    return uhr_suv

def get_mean_from_mask(pet_path: str, mask_path:str, structure_id: int|None = None) -> float:
    """Extracts mean intensity from a PET image using a FIRST segmentation mask.
    
    Args:
        pet_path (str): Path to the NIfTI PET image.
        mask_path (str): Path to the FIRST segmentation NIfTI image.
        structure_id (int, opyional): The ID of the brain structure (e.g., 10 for L_Thalamus).
        
    Returns:
        float: Mean intensity value, or np.nan if file/structure is missing.
    """
    if not os.path.exists(pet_path) or not os.path.exists(mask_path):
        return np.nan
        
    try:
        pet_obj = nib.load(pet_path)
        mask_obj = nib.load(mask_path)
        
        # Check if dimensions match
        if pet_obj.shape != mask_obj.shape:
            print(f"DEBUG: Dimension mismatch for {pet_path}. "
                  f"PET: {pet_obj.shape}, Mask: {mask_obj.shape}")
            return np.nan

        pet_data = pet_obj.get_fdata()
        mask_data = mask_obj.get_fdata()
        
        binary_mask = (mask_data == structure_id)
        mask_sum = np.sum(binary_mask)

        if mask_sum == 0:
            # This triggers if the ROI ID doesn't exist in the segmentation file
            print(f"DEBUG: ROI {structure_id} not found in mask for {pet_path}")
            return np.nan

        roi_data = pet_data[binary_mask]
        
        if np.all(np.isnan(roi_data)):
            print(f"DEBUG: All voxels in ROI {structure_id} are NaN in PET image")
            return np.nan

        # Use nanmean to be safe against isolated NaN voxels
        return np.nanmean(roi_data)
        
    except Exception as e:
        print(f"DEBUG: Unexpected error processing {pet_path}: {e}")
        return np.nan

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