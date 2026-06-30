import os
import argparse
import numpy as np
import nibabel as nib

def compute_relative_difference(test_data, ref_data, mask):
    """Computes voxel-wise relative difference between two images within a mask.

    The formula used is: ((Test - Reference) / Reference) * 100.
    Outliers are removed based on the 1.5 * median rule.

    Args:
        test_data (np.ndarray): Data from the image to be tested (e.g., sCT or TX).
        ref_data (np.ndarray): Data from the reference image (HRRT).
        mask (np.ndarray): Binary mask (1 for brain, 0 elsewhere).

    Returns:
        np.ndarray: The relative difference map (%) with outliers removed.
    """
    # Apply brain mask to isolate relevant voxels
    m_test = test_data * mask
    m_ref = ref_data * mask

    # Compute Relative Difference (RD) in %
    with np.errstate(divide='ignore', invalid='ignore'):
        rd = np.where(m_ref != 0, ((m_test - m_ref) / m_ref) * 100, np.nan)

    # Outlier removal (Edge errors) based on median-based thresholding
    median_rd = np.nanmedian(rd)
    lower_bound = 1.5 * median_rd
    upper_bound = -1.5 * median_rd

    threshold_mask = np.zeros(rd.shape)
    threshold_mask[(rd <= upper_bound) & (rd >= lower_bound)] = 1
    
    return rd * threshold_mask

def main():
    """Main execution function using argparse for NIfTI processing."""
    parser = argparse.ArgumentParser(
        description="Compute Relative Difference maps between a PET image and a Reference."
    )
    
    # Image Inputs
    parser.add_argument("--uhr", required=True, help="Path to the UHR PET image (e.g., DeepT1.nii.gz)")
    parser.add_argument("--hrrt", required=True, help="Path to the reference HRRT PET image (e.g., HRRTrecon.nii.gz)")
    
    # Mask Inputs (FSL FAST outputs)
    parser.add_argument("--pve1", required=True, help="Path to Grey Matter probability map (output_pve1.nii.gz)")
    parser.add_argument("--pve2", required=True, help="Path to White Matter probability map (output_pve2.nii.gz)")
    
    # Output Configuration
    parser.add_argument("--out_dir", required=True, help="Folder where the result will be saved")
    parser.add_argument("--sub_id", default="sub-01", help="Subject ID for the output filename")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Load Images
    print(f"--- Loading images for {args.sub_id} ---")
    try:
        uhr_img = nib.load(args.uhr)
        hrrt_img = nib.load(args.hrrt)
        gm_data = nib.load(args.pve1).get_fdata()
        wm_data = nib.load(args.pve2).get_fdata()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Create Whole Brain Mask (GM + WM > 10% probability)
    wbmask = ((gm_data > 0.1) | (wm_data > 0.1)).astype(float)

    # Compute Relative Difference
    print("--- Computing relative difference and removing outliers ---")
    rd_map = compute_relative_difference(uhr_img.get_fdata(), hrrt_img.get_fdata(), wbmask)

    # Save Result
    # Use reference image affine and header to ensure spatial consistency
    output_filename = f"reldiff_{args.sub_id}.nii.gz"
    output_path = os.path.join(args.out_dir, output_filename)
    
    result_img = nib.Nifti1Image(rd_map.astype(np.float32), hrrt_img.affine, hrrt_img.header)
    nib.save(result_img, output_path)
    
    print(f"--- Processing complete. File saved: {output_path} ---")

if __name__ == "__main__":
    main()