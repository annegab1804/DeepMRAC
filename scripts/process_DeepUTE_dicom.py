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
from deepmrac.predictions import predict_DeepUTE
from deepmrac.metrics import calculate_quality_metrics, save_metrics_to_csv

def run_pipeline(
    ute1_path: str,
    ute2_path: str,
    umap_path: str,
    output_folder: str,
    version: str = 'VE11P',
    save_prediction: bool | None = False,
    verbose: bool = True,
) -> None:
    """Executes the DeepUTE pipeline to generate MRAC DICOM files from UTE and Umap data.

    This implementation is based on the methodology described in:
    Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for 
    brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the 
    training group size. Neuroimage. 2020;222:117221. doi:10.1016/j.neuroimage.2020.117221

    The pipeline sorts DICOM files for both UTE echoes, converts them to NIfTI, 
    performs isotropic resampling, runs the DeepUTE prediction model (dual-channel), 
    and exports the final result back into DICOM format using the Umap as a template. 
    It concludes by calculating quality metrics (MAE, PSNR, SSIM, Dice).

    Args:
        ute1_path: Path to the directory containing UTE Echo 1 DICOM files.
        ute2_path: Path to the directory containing UTE Echo 2 DICOM files.
        umap_path: Path to the directory containing Umap (template) DICOM files.
        output_folder: Path where the resulting MRAC DICOM files will be saved.
        version: Model training version to use (e.g., 'VB20P' or 'VE11P'). 
            Defaults to 'VE11P'.
        save_prediction: If True, saves the resampled volume as 'DeepUTE_QC.nii.gz' 
            for quality control. Defaults to False.
        verbose: If True, prints progress and status messages to the console. 
            Defaults to True.

    Returns:
        None. The generated DICOM files are saved directly to `output_folder`.

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
        # Sort files into specific folders
        sort_files(source_folder=ute1_path, temp_subfolder=f"{tmpdir}/ute1_dcm", verbose=verbose )
        sort_files(source_folder=ute2_path, temp_subfolder=f"{tmpdir}/ute2_dcm", verbose=verbose )
        sort_files(source_folder=umap_path, temp_subfolder=f"{tmpdir}/umap_dcm", verbose=verbose )

        convert_to_nifti(dicom_dir=f"{tmpdir}/ute1_dcm", output_nii=f"{tmpdir}/ute1.nii.gz", verbose=verbose)
        convert_to_nifti(dicom_dir=f"{tmpdir}/ute2_dcm", output_nii=f"{tmpdir}/ute2.nii.gz", verbose=verbose)
        convert_to_nifti(dicom_dir=f"{tmpdir}/umap_dcm", output_nii=f"{tmpdir}/umap.nii.gz", verbose=verbose)
        
        # Load and resample data
        ute1_rsl, ute1_ref = load_and_resample_images(nii_image=f"{tmpdir}/ute1.nii.gz", verbose=verbose)
        ute2_rsl, ute2_ref = load_and_resample_images(nii_image=f"{tmpdir}/ute2.nii.gz", verbose=verbose)
        umap_nat = nib.load(f'{tmpdir}/umap.nii.gz')
        
        # Predict
        pred = predict_DeepUTE(ute1=ute1_rsl, ute2=ute2_rsl, version=version)

        # Resample to Umap Format
        DeepX = np.flip(pred, axis=1)
        DeepX = resample_to_output_format(pred=DeepX, model_ref=ute1_ref, umap_native=umap_nat, rmi_type="UTE", verbose=verbose, save_prediction=save_prediction)
        
        # Final DICOM (using Umap as container)
        DeepX = np.transpose(DeepX, (2, 1, 0))
        to_dcm(DeepX=DeepX, dcmcontainer=f"{tmpdir}/umap_dcm", dicomfolder=output_folder, rmi_type="UTE")
        
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

        save_metrics_to_csv(metrics_dict=metrics_dict, rmi_type='UTE', output_folder=f"{output_folder}/metrics")


    finally: 
        # Cleanup
        shutil.rmtree(tmpdir)

def main():
    """
    MRAC prediction using Deep Learning 3D U-net
    Author: Claes Ladefoged, Rigshospitalet, Copenhagen, Denmark
            claes.noehr.ladefoged@regionh.dk
    Version: March-12-2019
    """
    parser = argparse.ArgumentParser(description='Predict using DeepUTE.')
    parser.add_argument(
        "--ute1_path", 
        help="Path to folder with dicom files of UTE Echo 1.", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--ute2_path", 
        help="Path to folder with dicom files of UTE Echo 2.", 
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
        ute1_path=args.ute1_path,
        ute2_path=args.ute2_path,
        umap_path=args.umap_path,
        output_folder=args.output_folder,
        version=args.version,
        save_prediction=args.save_prediction,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()