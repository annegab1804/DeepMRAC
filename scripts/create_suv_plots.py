import os
import argparse
import numpy as np
import ants
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
import xml.etree.ElementTree as ET

from deepmrac.suv_utils import generate_summary_bland_altman, generate_summary_percentage_difference, generate_summary_violin

def create_suv_plots(
    patient_dir: str,
    output_dir: str,
    uhr_name: str,
    hrrt_name: str,
    t1_name: str,
) -> None :
    """Orchestrates the extraction of ROI means and generates cohort-level SUV plots.

    Workflow:
    1. Iterates through patient folders in patient_dir.
    2. Segments T1 and registers T1 to MNI space using ANTs (SyN).
    3. Warps UHR and HRRT PET images to MNI space.
    4. Normalizes PET intensities by the mean of the Gray/White matter mask (SUVR).
    5. Extracts ROI means using the AAL atlas.
    6. Generates summary Violin, Bland-Altman, and % Difference plots.

    Args:
        patient_dir: Root directory containing patient sub-folders.
        output_dir: Path where the resulting summary PNG files will be saved.
        uhr_name: Filename of the UHR PET image (e.g., 'uhr_suv.nii.gz').
        hrrt_name: Filename of the HRRT PET image (e.g., 'hrrt_suv.nii.gz').
        t1_name: Filename of the T1 weighted nifti image (e.g., 't1.nii.gz').
    """
    # Find the Atlas
    aal_dir = os.path.expanduser('~/nilearn_data/aal_SPM12/')
    # We need the .nii file and the .txt or .xml labels
    aal_atlas_path = os.path.join(aal_dir, 'aal/ROI_MNI_V4.nii')
    labels_path = os.path.join(aal_dir, 'aal/ROI_MNI_V4.xml')

    # Define the regions of interest you want
    target_names = {
        'Hippocampus_L', 'Hippocampus_R',
        'Caudate_L', 'Caudate_R',
        'Putamen_L', 'Putamen_R',
        'Pallidum_L', 'Pallidum_R',
        'Thalamus_L', 'Thalamus_R',
        'Amygdala_L', 'Amygdala_R'
    }

    tree = ET.parse(labels_path)
    root = tree.getroot()

    aal_roi_configs = {}
    for label in root.iter('label'):
        index = label.find('index')
        name  = label.find('name')
        if index is not None and name is not None and name.text in target_names:
            aal_roi_configs[int(index.text)] = name.text

    print("Loaded ROI configs:", aal_roi_configs)

    cohort_results = {name: {'UHR': [], 'HRRT': []} for name in aal_roi_configs.values()}
    mni_template = ants.image_read(ants.get_ants_data('mni'))

    # Get all items, join path, and filter to keep only directories
    patient_dirs = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir) 
                    if os.path.isdir(os.path.join(patient_dir, d))]

    for p_dir in patient_dirs:
        sub_id = os.path.basename(p_dir)
        print(f"--- Processing {sub_id} ---")

        # Paths
        t1_path = os.path.join(p_dir, t1_name)
        uhr_path = os.path.join(p_dir, uhr_name)
        hrrt_path = os.path.join(p_dir, hrrt_name)
        
        # Load Images
        t1 = ants.image_read(t1_path)
        uhr = ants.image_read(uhr_path)
        hrrt = ants.image_read(hrrt_path)

        # Registration: T1 -> MNI
        reg = ants.registration(fixed=mni_template, moving=t1, type_of_transform='SyN')
        
        # Tissue Segmentation (for Whole Brain Mask)
        t1_mask = ants.get_mask(t1)

        # N3 bias correction first (improves segmentation quality)
        t1_n3 = ants.n3_bias_field_correction(t1)

        # Atropos: 3-class segmentation (1=CSF, 2=GM, 3=WM)
        seg = ants.atropos(
            a=t1_n3,
            m='[0.2,1x1x1]',
            c='[3,0]',
            i='kmeans[3]',
            x=t1_mask
        )

        # Warp masks to MNI space
        gm_mni = ants.apply_transforms(fixed=mni_template, moving=seg['probabilityimages'][1],
                                        transformlist=reg['fwdtransforms'])
        wm_mni = ants.apply_transforms(fixed=mni_template, moving=seg['probabilityimages'][2],
                                        transformlist=reg['fwdtransforms'])
        
        # Warp PET images to MNI
        uhr_mni = ants.apply_transforms(fixed=mni_template, moving=uhr, transformlist=reg['fwdtransforms'])
        hrrt_mni = ants.apply_transforms(fixed=mni_template, moving=hrrt, transformlist=reg['fwdtransforms'])

        # Intensity Normalization (SUVR)
        # Create Whole Brain mask in MNI space (Threshold 0.5)
        wb_mask = (gm_mni + wm_mni).numpy() > 0.5
        
        uhr_data = uhr_mni.numpy()
        hrrt_data = hrrt_mni.numpy()
        
        uhr_suvr_data = uhr_data / np.mean(uhr_data[wb_mask])
        hrrt_suvr_data = hrrt_data / np.mean(hrrt_data[wb_mask])

        # Convert back to Nibabel for Nilearn extraction
        uhr_nii  = ants.to_nibabel_nifti(uhr_mni)
        hrrt_nii = ants.to_nibabel_nifti(hrrt_mni)
        uhr_nii  = nib.Nifti1Image(uhr_suvr_data, uhr_nii.affine, uhr_nii.header)
        hrrt_nii = nib.Nifti1Image(hrrt_suvr_data, hrrt_nii.affine, hrrt_nii.header)

        # ROI Extraction using AAL
        masker = NiftiLabelsMasker(labels_img=aal_atlas_path, resampling_target="data")
        
        # Get mean for all AAL regions
        uhr_means = masker.fit_transform(uhr_nii).flatten() 
        hrrt_means = masker.fit_transform(hrrt_nii).flatten()

        # Map specific ROI IDs to our results dict
        # Note: AAL IDs in the NIfTI usually start from 1, Nilearn's output follows the sorted label order
        labels = masker.labels_ 
        for atlas_id, roi_name in aal_roi_configs.items():
            if atlas_id in labels:
                idx = labels.index(atlas_id)
                cohort_results[roi_name]['UHR'].append(uhr_means[idx])
                cohort_results[roi_name]['HRRT'].append(hrrt_means[idx])
    
    # Visualizations
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    generate_summary_violin(cohort_results, os.path.join(output_dir, "Violin_SUVR.png"))
    generate_summary_bland_altman(cohort_results, os.path.join(output_dir, "BlandAltman_SUVR.png"))
    generate_summary_percentage_difference(cohort_results, os.path.join(output_dir, "Diff_SUVR.png"))
            

def main():
    parser = argparse.ArgumentParser(description='Cohort PET Analysis using Bland-Altman.')
    
    # Paths
    parser.add_argument("--patient_dir", required=True, help="Root folder containing patient sub-folders.")
    parser.add_argument("--output_dir", required=True, help="Folder to save the plots.")
    
    # Dynamic Filenames
    parser.add_argument("--uhr_name", required=True, help="Name of the UHR PET file (e.g., DeepT1.nii.gz)")
    parser.add_argument("--hrrt_name", required=True, help="Name of the reference HRRT PET file (e.g., HRRTrecon.nii.gz)")
    parser.add_argument("--t1_name", required=True, help="Name of the T1 weighted file (e.g., T1.nii.gz).")

    args = parser.parse_args()

    create_suv_plots(
        patient_dir=args.patient_dir,
        output_dir=args.output_dir,
        uhr_name=args.uhr_name,
        hrrt_name=args.hrrt_name,
        t1_name=args.t1_name,
    )

if __name__ == "__main__":
    main()