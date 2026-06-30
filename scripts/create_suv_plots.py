import os
import argparse
import numpy as np
import ants

from deepmrac.suv_utils import generate_summary_bland_altman, generate_summary_percentage_difference, generate_summary_violin, align_pet_to_t1, create_whole_brain_mask, extracting_roi

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
    2. Aligns PET images (UHR/HRRT) to their respective T1 structural space (Native).
    3. Computes T1-to-MNI non-linear registration (SyN) and generates a 
       Whole Brain mask in MNI space.
    4. Calculates SUVR normalization factors by sampling MNI-warped PET 
       images within the Whole Brain mask.
    5. Warps the AAL atlas to Native T1 space to extract mean ROI signals.
    6. Normalizes extracted values by the SUVR factor and generates summary 
       Violin, Bland-Altman, and % Difference plots.

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
    mni_template = ants.image_read(ants.get_ants_data('mni'))

    # Define the regions of interest you want
    target_names = {
        'Hippocampus_L', 'Hippocampus_R',
        'Caudate_L', 'Caudate_R',
        'Putamen_L', 'Putamen_R',
        'Pallidum_L', 'Pallidum_R',
        'Thalamus_L', 'Thalamus_R',
        'Amygdala_L', 'Amygdala_R'
    }

    cohort_results = {name: {'UHR': [], 'HRRT': []} for name in target_names}

    # Iterate through patients
    patient_dirs = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir) 
                    if os.path.isdir(os.path.join(patient_dir, d))]

    for p_dir in patient_dirs:
        sub_id = os.path.basename(p_dir)
        print(f"--- Processing {sub_id} ---")

        # Paths
        t1_path = os.path.join(p_dir, t1_name)
        uhr_path = os.path.join(p_dir, uhr_name)
        hrrt_path = os.path.join(p_dir, hrrt_name)

        # Temp file for aligned PET (as required by current extracting_roi signature)
        uhr_aligned_path = os.path.join(p_dir, "uhr_in_t1.nii.gz")
        hrrt_aligned_path = os.path.join(p_dir, "hrrt_in_t1.nii.gz")

        # STEP 1: Align PET to T1 (Intra-subject)
        align_pet_to_t1(uhr_path, t1_path, os.path.dirname(uhr_aligned_path))
        align_pet_to_t1(hrrt_path, t1_path, os.path.dirname(hrrt_aligned_path))

        # STEP 2: Register T1 to MNI for Masking (SyN for accuracy)
        t1 = ants.image_read(t1_path)
        reg_t1_mni = ants.registration(fixed=mni_template, moving=t1, type_of_transform='SyN')

        # STEP 3: Create Whole Brain Mask
        wb_mask_mni = create_whole_brain_mask(t1_path, reg_t1_mni)

        # STEP 4: Intensity Normalization (SUVR)
        # Note: We warp PET to MNI just to calculate the SUVR reference
        uhr_mni = ants.apply_transforms(fixed=mni_template, moving=ants.image_read(uhr_aligned_path), 
                                        transformlist=reg_t1_mni['fwdtransforms'])
        hrrt_mni = ants.apply_transforms(fixed=mni_template, moving=ants.image_read(hrrt_aligned_path), 
                                         transformlist=reg_t1_mni['fwdtransforms'])
        
        uhr_data = uhr_mni.numpy()
        hrrt_data = hrrt_mni.numpy()

        # Use the mask to get mean and normalize
        uhr_suvr_val = np.mean(uhr_data[wb_mask_mni])
        hrrt_suvr_val = np.mean(hrrt_data[wb_mask_mni])

        # STEP 5: ROI Extraction 
        # This function handles the warp of the atlas internally
        uhr_rois = extracting_roi(t1_path, uhr_aligned_path, aal_atlas_path, labels_path, target_names)
        hrrt_rois = extracting_roi(t1_path, hrrt_aligned_path, aal_atlas_path, labels_path, target_names)

        # STEP 6: Store results (Normalizing the means by the SUVR reference)
        for roi in target_names:
            if roi in uhr_rois and roi in hrrt_rois:
                cohort_results[roi]['UHR'].append(uhr_rois[roi] / uhr_suvr_val)
                cohort_results[roi]['HRRT'].append(hrrt_rois[roi] / hrrt_suvr_val)

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