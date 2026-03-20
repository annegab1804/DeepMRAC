import argparse
import tempfile
import shutil
import os
import nibabel as nib
import numpy as np
from deepmrac.utils import (
    str2bool,
    sort_dicomfiles,
    convert_dicom_to_nifti,
    convert_interfile_to_nifti,
    load_and_resample_images,
    resample_to_output_format,
    to_dcm,
    to_interfile
)
from deepmrac.predictions import predict_DeepT1
from deepmrac.metrics import calculate_quality_metrics, save_metrics_to_csv
from deepmrac.plots import plot_3d_views, load_nifti_volume_and_aspects

def run_pipeline(
    t1_path: str,
    umap_path: str,
    output_folder: str,
    ct_path: str | None = None,
    version: str | None = 'VE11P',
    verbose: bool | None = True,
):
    """Executes the DeepT1 pipeline to generate MRAC DICOM files from T1 and Umap data.

    This implementation is based on the methodology described in:
    Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for 
    brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the 
    training group size. Neuroimage. 2020;222:117221. doi:10.1016/j.neuroimage.2020.117221

    This pipeline sorts DICOM inputs, converts them to NIfTI, performs resampling, 
    runs the deep learning prediction model, and exports the final result back 
    into a DICOM format using the Umap as a header template. It finally calculates
    and prints quality metrics (MAE, PSNR, SSIM, Dice).

    Args:
        t1_path: Path to the directory containing T1-weighted MPRAGE DICOM files.
        umap_path: Path to the directory containing Umap DICOM or interfile files.
        output_folder: Path where the resulting MRAC DICOM files will be saved.
        ct_path: Path to the directory containing original CT DICOM files. 
            Defaults to None.
        version: Model training version to use (e.g., 'VB20P' or 'VE11P'). 
            Defaults to 'VE11P'.
        save_prediction: If True, saves the resampled volume as 'DeepT1_QC.nii.gz' 
            for quality control. Defaults to False.
        verbose: If True, prints progress and status messages to the console. 
            Defaults to True.

    Returns:
        None. The function saves the generated DICOM files directly to the 
        `output_folder`.

    Raises:
        FileExistsError: If the `output_folder` already exists and contains files, 
            or if the temporary sorting directory is not empty.
        FileNotFoundError: If any of the input paths do not exist.
        AttributeError: If a DICOM file is encountered that lacks an 'InstanceNumber'.
        RuntimeError: If two DICOM files share the same InstanceNumber, or if 
            NIfTI conversion/model prediction fails.
    """
    if os.path.exists(output_folder) and os.listdir(output_folder):
        raise FileExistsError(
            f"The output folder '{output_folder}' already exists and is not empty. "
            "Please delete it or provide a different path to avoid data contamination."
        )
    
    # Create temporary folder    
    tmpdir = tempfile.mkdtemp()

    try:
        # Sort and convert files into specific folders
        # --- Process T1 (Standard DICOM) ---
        sort_dicomfiles(source_folder=t1_path, temp_subfolder=f"{tmpdir}/t1_dcm", verbose=verbose)
        convert_dicom_to_nifti(dicom_dir=f"{tmpdir}/t1_dcm", output_nii=f"{tmpdir}/t1.nii.gz", verbose=verbose)

        # --- Process UMAP (Conditional: DICOM or Interfile) ---
        # Check if there's an Interfile header in the source folder
        interfile_headers = [f for f in os.listdir(umap_path) if f.lower().endswith('.i.hdr')]

        if interfile_headers:
            if verbose:
                print(f"Detected Interfile format for UMAP in {umap_path}")
            
            # We take the first header found
            hdr_full_path = os.path.join(umap_path, interfile_headers[0])

            convert_interfile_to_nifti(hdr_path=hdr_full_path, output_nii_path=f"{tmpdir}/umap.nii.gz")

        else:
            if verbose:
                print(f"Detected DICOM format for UMAP in {umap_path}")
                
            # Standard sorting for DICOM files
            sort_dicomfiles(source_folder=umap_path, temp_subfolder=f"{tmpdir}/umap_dcm", verbose=verbose)
            convert_dicom_to_nifti(dicom_dir=f"{tmpdir}/umap_dcm", output_nii=f"{tmpdir}/umap.nii.gz", verbose=verbose)
        
        # Load and ressample data
        t1_rsl, t1_ref = load_and_resample_images(nii_image=f"{tmpdir}/t1.nii.gz", verbose=verbose)
        umap_nat = nib.load(f'{tmpdir}/umap.nii.gz')

        # Flip to match orientation on what was trained on
        t1_rsl = np.flipud(np.swapaxes(t1_rsl, 0, 2)) 

        # Predict
        pred = predict_DeepT1(t1=t1_rsl, version=version)

        # Flip back to the original orientation
        pred = np.swapaxes(np.flipud(pred), 2, 0)

        # Resample to Umap Formatos.makedirs(output_folder, exist_ok=True)
        pred_nii = nib.Nifti1Image(pred, t1_ref.affine, t1_ref.header)
        os.makedirs(output_folder, exist_ok=True)
        DeepX = resample_to_output_format(pred_nii=pred_nii, umap_native=umap_nat, verbose=verbose, output_file=f"{output_folder}/DeepT1.nii.gz")
       
        # Final DICOM (using Umap as container)
        if interfile_headers:
            to_interfile(DeepX=DeepX, hdr_template=hdr_full_path, output_path=f"{output_folder}/sCT", rmi_type="T1")
        else:
            to_dcm(DeepX=DeepX, dcmcontainer=f"{tmpdir}/umap_dcm", dicomfolder=f"{output_folder}/sCT", rmi_type="T1")
                
        print(f"Success! Result saved in: {output_folder}")

        # Plot
        volume, aspects = load_nifti_volume_and_aspects(nifti_path=f"{output_folder}/DeepT1.nii.gz")
        plot_3d_views(volume=volume, aspects=aspects, rotation_map=None, flip_map=None)


        if ct_path and os.path.exists(ct_path):
            is_nifti = ct_path.lower().endswith(('.nii', '.nii.gz'))

            if is_nifti:
                if verbose: print(f"Input is already NIfTI: {ct_path}")
                ct_nat = nib.load(ct_path)
            else:
                # DICOM files
                sort_dicomfiles(source_folder=ct_path, temp_subfolder=f"{tmpdir}/ct_dcm", verbose=verbose)
                convert_dicom_to_nifti(dicom_dir=f"{tmpdir}/ct_dcm", output_nii=f"{tmpdir}/ct.nii.gz", verbose=verbose)
                ct_nat = nib.load(f'{tmpdir}/ct.nii.gz')
            

            # Resample using Umap's grid size
            ct_rsl = resample_to_output_format(pred_nii=ct_nat, umap_native=umap_nat, verbose=verbose, output_file=f"{output_folder}/CT_resampled.nii.gz")

            # Make sure we have the same units as the sCT (min=0)
            ct_final_data = np.clip(ct_rsl, a_min=-1000, a_max=None)
            ct_final_data = ct_final_data + 1000

            # Final DICOM (using Umap as container)
            ct_final_data = np.transpose(ct_final_data, (1, 2, 0))

            to_dcm(DeepX=ct_final_data, dcmcontainer=f"{tmpdir}/umap_dcm",  dicomfolder=f"{output_folder}/CT", rmi_type="CT")

            # Calculate metrics
            metrics_dict = calculate_quality_metrics(
                sct_path=f"{output_folder}/sCT",
                ct_path=f"{output_folder}/CT",
            )
            if verbose:
                print("\n" + "="*30)
                print(" GLOBAL QUALITY METRICS ")
                print("="*30)
                print(f"PSNR: {metrics_dict['PSNR']:.2f}")
                print(f"SSIM: {metrics_dict['SSIM']:.4f}")
                
                for mode in ['tissue', 'bone']:
                    print(f"\n--- {mode.upper()} ANALYSIS ---")
                    print(f"MAE:  {metrics_dict[f'{mode}_MAE']:.4f}")
                    print(f"ME:   {metrics_dict[f'{mode}_ME']:.4f}")
                    print(f"RE:   {metrics_dict[f'{mode}_RE']:.4f}")
                    print(f"ARE:  {metrics_dict[f'{mode}_ARE']:.4f}")
                    print(f"Dice: {metrics_dict[f'{mode}_Dice']:.4f}")
                print("="*30)

            save_metrics_to_csv(metrics_dict=metrics_dict, rmi_type='T1', output_folder=f"{output_folder}/metrics")

    finally:
        shutil.rmtree(tmpdir)

def main():
    """MRAC prediction using Deep Learning 3D U-net
    Author: Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
            claes.noehr.ladefoged@regionh.dk
    Version: August-20-2019
    """
    parser = argparse.ArgumentParser(description='Predict using DeepDixon.')
    parser.add_argument(
        "--t1_path", 
        help="Path to folder with dicom files of T1 weighted MPRAGE.", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--umap_path", 
        help="Path to folder with dicom or interfile files of Umap.", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--ct_path", 
        help="Path to folder with dicom files of original CT.", 
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_folder", 
        help="Name for output folder. ", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--version", 
        help="Software version used to train the model (VB20P or VE11P) Default: VE11P. ",
        type=str,
        default='VE11P'
    )
    parser.add_argument(
        "--verbose", 
        type=str2bool, 
        default=False
    )
    args = parser.parse_args()

    run_pipeline(
        t1_path=args.t1_path,
        umap_path=args.umap_path,
        ct_path=args.ct_path,
        output_folder=args.output_folder,
        version=args.version,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()