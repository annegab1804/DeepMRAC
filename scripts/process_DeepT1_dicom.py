import argparse
import tempfile
import shutil
import os
import nibabel as nib
import numpy as np
from deepmrac.utils import (
    sort_files,
    convert_to_nifti,
    load_and_resample_images,
    resample_to_output_format,
    to_dcm
)
from deepmrac.predictions import predict_DeepT1
from deepmrac.metrics import calculate_quality_metrics, save_metrics_to_csv

def run_pipeline(
    t1_path: str,
    umap_path: str,
    output_folder: str,
    version: str | None = 'VE11P',
    save_prediction: bool | None = False,
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
        umap_path: Path to the directory containing Umap DICOM files.
        output_folder: Path where the resulting MRAC DICOM files will be saved.
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
        sort_files(source_folder=t1_path, temp_subfolder=f"{tmpdir}/t1_dcm", verbose=verbose)
        sort_files(source_folder=umap_path, temp_subfolder=f"{tmpdir}/umap_dcm",verbose=verbose)

        convert_to_nifti(dicom_dir=f"{tmpdir}/t1_dcm", output_nii=f"{tmpdir}/t1.nii.gz", verbose=verbose)
        convert_to_nifti(dicom_dir=f"{tmpdir}/umap_dcm", output_nii=f"{tmpdir}/umap.nii.gz", verbose=verbose)
        
        # Load and ressample data
        t1_rsl, t1_ref = load_and_resample_images(nii_image=f"{tmpdir}/t1.nii.gz", verbose=verbose)
        umap_nat = nib.load(f'{tmpdir}/umap.nii.gz')

        # Flip to match orientation on what was trained on
        t1_rsl = np.flipud(np.swapaxes(t1_rsl, 0, 2)) 

        # Predict
        pred = predict_DeepT1(t1=t1_rsl, version=version)

        # Flip back to the original orientation
        pred = np.swapaxes(np.flipud(pred), 2, 0)

        # Resample to Umap Format
        DeepX = resample_to_output_format(pred=pred, model_ref=t1_ref, umap_native=umap_nat, rmi_type="T1", verbose=verbose, save_prediction=save_prediction)
       
        # Final DICOM (using Umap as container)
        DeepX = np.transpose(DeepX, (1, 2, 0))
        DeepX = np.flip(DeepX, axis=0)
        DeepX = np.flip(DeepX, axis=1)
        to_dcm(DeepX=DeepX, dcmcontainer=f"{tmpdir}/umap_dcm", dicomfolder=output_folder, rmi_type="T1")
        
        print(f"Success! Result saved in: {output_folder}")

        # Calculate metrics
        metrics_dict = calculate_quality_metrics(
            sct_path=output_folder,
            umap_path=f"{tmpdir}/umap_dcm",
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
        help="Path to folder with dicom files of Umap.", 
        type=str,
        required=True
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
        "--save_prediction",
        help="If True, saves the resampled volume as DeepT1_QC.nii.gz for quality control. Defaults to False.",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    run_pipeline(
        t1_path=args.t1_path,
        umap_path=args.umap_path,
        output_folder=args.output_folder,
        version=args.version,
        save_prediction=args.save_prediction,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()