import os
import argparse
import numpy as np

from deepmrac.suv_utils import get_mean_from_mask, generate_summary_bland_altman, generate_summary_percentage_difference, generate_summary_violin

def create_suv_plots(
    input_folder: str,
    output_folder: str,
    uhr_name: str,
    hrrt_name: str,
    seg_name: str = "output_all_fast_firstseg.nii.gz"
) -> None :
    """Orchestrates the extraction of ROI means and generates cohort-level SUV plots.

    This function iterates through patient subdirectories within an input folder,
    extracts the mean SUV values for a predefined set of ROIs (FSL FIRST IDs),
    and generates three summary visualizations: Violin plots, Bland-Altman plots,
    and a percentage difference bar chart.

    input_folder/
    ├── patient_01/
    │   ├── uhr_suv.nii.gz   <-- uhr_name
    │   ├── hrrt_suv.nii.gz  <-- hrrt_name
    │   └── output_all_fast_firstseg.nii.gz  <-- seg_name
    ├── patient_02/
    │   ├── uhr_suv.nii.gz
    │   ├── hrrt_suv.nii.gz
    │   └── output_all_fast_firstseg.nii.gz
    └── patient_03/
        ├── uhr_suv.nii.gz
        ├── hrrt_suv.nii.gz
        └── output_all_fast_firstseg.nii.gz

    Args:
        input_folder: Path to the root directory containing patient sub-folders.
            Each sub-folder must contain the UHR, HRRT, and segmentation files.
        output_folder: Path where the resulting summary PNG files will be saved.
        uhr_name: Filename of the UHR PET image (e.g., 'uhr_suv.nii.gz').
        hrrt_name: Filename of the HRRT PET image (e.g., 'hrrt_suv.nii.gz').
        seg_name: Filename of the segmentation mask (default: FIRST segmentation).

    Returns:
        None. Saves "SUMMARY_Violin_Plots.png", "SUMMARY_BlandAltman_Plots.png",
        and "SUMMARY_Percentage_Difference.png" to the output_folder.

    Raises:
        OSError: If input_folder does not exist or output_folder cannot be created.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ROI IDs from FSL FIRST
    roi_configs = {
        # Thalamus
        10: 'L_Thalamus', 49: 'R_Thalamus',
        # Noyau Caudé
        11: 'L_Caudate', 50: 'R_Caudate',
        # Putamen
        12: 'L_Putamen', 51: 'R_Putamen',
        # Pallidum
        13: 'L_Pallidum', 52: 'R_Pallidum',
        # Hippocampe
        17: 'L_Hippocampus', 53: 'R_Hippocampus',
        # Amygdale
        18: 'L_Amygdala', 54: 'R_Amygdala',
        # Accumbens
        26: 'L_Accumbens', 58: 'R_Accumbens',
        # Tronc Cérébral
        16: 'Brain_Stem'
    }
    cohort_results = {name: {'UHR': [], 'HRRT': []} for name in roi_configs.values()}

    # Get all items, join path, and filter to keep only directories
    patient_folders = sorted([
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if os.path.isdir(os.path.join(input_folder, f))
    ])

    print(f"Found {len(patient_folders)} subfolders.")
        
    for p_dir in patient_folders:
        uhr_path = os.path.join(p_dir, uhr_name)
        hrrt_path = os.path.join(p_dir, hrrt_name)
        seg_path = os.path.join(p_dir, seg_name)

        for roi_id, roi_name in roi_configs.items():
            val_uhr = get_mean_from_mask(uhr_path, seg_path, roi_id)
            val_hrrt = get_mean_from_mask(hrrt_path, seg_path, roi_id)
            
            if not np.isnan(val_uhr) and not np.isnan(val_hrrt):
                cohort_results[roi_name]['UHR'].append(val_uhr)
                cohort_results[roi_name]['HRRT'].append(val_hrrt)

    output_violin = os.path.join(output_folder, "SUMMARY_Violin_Plots.png")
    generate_summary_violin(cohort_results, output_violin)
    
    output_ba = os.path.join(output_folder, "SUMMARY_BlandAltman_Plots.png")
    generate_summary_bland_altman(cohort_results, output_ba)

    output_diff = os.path.join(output_folder, "SUMMARY_Percentage_Difference.png")
    generate_summary_percentage_difference(cohort_results, output_diff)


def main():
    parser = argparse.ArgumentParser(description='Cohort PET Analysis using Bland-Altman.')
    
    # Paths
    parser.add_argument("--input_folder", required=True, help="Root folder containing patient sub-folders.")
    parser.add_argument("--output_folder", required=True, help="Folder to save the plots.")
    
    # Dynamic Filenames
    parser.add_argument("--uhr_name", required=True, help="Name of the UHR PET file (e.g., DeepT1.nii.gz)")
    parser.add_argument("--hrrt_name", required=True, help="Name of the reference HRRT PET file (e.g., HRRTrecon.nii.gz)")
    parser.add_argument("--seg_name", default="output_all_fast_firstseg.nii.gz", help="Name of the FIRST segmentation file.")

    args = parser.parse_args()

    create_suv_plots(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        uhr_name=args.uhr_name,
        hrrt_name=args.hrrt_name,
        seg_name=args.seg_name
    )

if __name__ == "__main__":
    main()