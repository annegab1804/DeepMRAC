import argparse
import tempfile
import shutil
import nibabel as nib
import numpy as np
from deepmrac.utils import (
    sort_files,
    convert_to_nifti,
    load_and_resample_images,
    resample_to_output_format,
    to_dcm
)
from deepmrac.predictions import predict_DeepDixon
from deepmrac.metrics import calculate_quality_metrics, save_metrics_to_csv

def run_pipeline(
    inphase_path: str,
    opposedphase_path: str,
    umap_path: str,
    output_folder: str,
    version: str = 'VE11P',
    save_prediction: bool | None = False,
    verbose: bool = True,
) -> None:
    """Executes the DeepDixon pipeline to generate MRAC DICOM files from Dixon and Umap data.

    This implementation is based on the methodology described in:
    Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for 
    brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the 
    training group size. Neuroimage. 2020;222:117221. doi:10.1016/j.neuroimage.2020.117221

    The pipeline sorts DICOM files for both Dixon phases (In-phase and Opposed-phase), 
    converts them to NIfTI, performs isotropic resampling to 192^3, runs the 
    DeepDixon prediction model (dual-channel), and exports the final result back 
    into DICOM format using the Umap as a template. It concludes by calculating 
    quality metrics (MAE, PSNR, SSIM, Dice).

    Args:
        inphase_path: Path to the directory containing Dixon In-phase DICOM files.
        opposedphase_path: Path to the directory containing Dixon Opposed-phase DICOM files.
        umap_path: Path to the directory containing Umap (template) DICOM files.
        output_folder: Path where the resulting MRAC DICOM files will be saved.
        version: Model training version to use (e.g., 'VB20P' or 'VE11P'). 
            Defaults to 'VE11P'.
        save_prediction: If True, saves the resampled volume as 'DeepDixon_QC.nii.gz' 
            for quality control. Defaults to False.
        verbose: If True, prints progress and status messages to the console. 
            Defaults to True.

    Returns:
        None. The generated DICOM files are saved directly to `output_folder`.

    Raises:
        FileNotFoundError: If any of the input paths do not exist.
        RuntimeError: If NIfTI conversion or model prediction fails.
    """
    # Create temporary folder    
    tmpdir = tempfile.mkdtemp()

    try:

        # Sort and convert files into specific folders
        sort_files(source_folder=inphase_path, temp_subfolder=f"{tmpdir}/inphase_dcm", verbose=verbose )
        sort_files(source_folder=opposedphase_path, temp_subfolder=f"{tmpdir}/opposedphase_dcm", verbose=verbose )
        sort_files(source_folder=umap_path, temp_subfolder=f"{tmpdir}/umap_dcm", verbose=verbose )

        convert_to_nifti(dicom_dir=f"{tmpdir}/inphase_dcm", output_nii=f"{tmpdir}/inphase.nii.gz", verbose=verbose)
        convert_to_nifti(dicom_dir=f"{tmpdir}/opposedphase_dcm", output_nii=f"{tmpdir}/opposedphase.nii.gz", verbose=verbose)
        convert_to_nifti(dicom_dir=f"{tmpdir}/umap_dcm", output_nii=f"{tmpdir}/umap.nii.gz", verbose=verbose)
        
        # Load and resample data
        inphase_rsl, inphase_ref = load_and_resample_images(nii_image=f"{tmpdir}/inphase.nii.gz", verbose=verbose)
        opposedphase_rsl, opposedphase_ref = load_and_resample_images(nii_image=f"{tmpdir}/opposedphase.nii.gz", verbose=verbose)
        umap_nat = nib.load(f'{tmpdir}/umap.nii.gz')
    
        # Predict
        pred = predict_DeepDixon(inphase=inphase_rsl, opposedphase=opposedphase_rsl, version=version)

        # Resample to Umap Format
        DeepX = np.flip(pred, axis=2)
        DeepX = resample_to_output_format(pred=DeepX, model_ref=inphase_ref, umap_native=umap_nat, rmi_type="Dixon", verbose=verbose, save_prediction=save_prediction)

        # Final DICOM (using Umap as container)
        DeepX = np.transpose(DeepX, (1, 2, 0))
        to_dcm(DeepX=DeepX, dcmcontainer=f"{tmpdir}/umap_dcm", dicomfolder=output_folder, rmi_type="Dixon")
        
        print(f"Success! Result saved in: {output_folder}")

        # Calculate metrics
        metrics_dict = calculate_quality_metrics(
            sct_path=output_folder,
            umap_path=f"{tmpdir}/umap_dcm",
            dice_threshold=300
        )

        print(f"Mean absolut error (MAE)): {metrics_dict['MAE']}")
        print(f"Peek signal to noise ratio (PSNR): {metrics_dict['PSNR']}")
        print(f"Structural Similarity Index Measure (SSIM) : {metrics_dict['SSIM']}")
        print(f"Dice similarity coefficient (DSC): {metrics_dict['Dice']}")

        save_metrics_to_csv(metrics_dict=metrics_dict, rmi_type='Dixon', output_folder=f"{output_folder}/metrics")


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
        "--inphase_path", 
        help="Path to folder with dicom files of Dixon in-phase.", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--opposedphase_path", 
        help="Path to folder with dicom files of Dixon opposed-phase.", 
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
        inphase_path=args.inphase_path,
        opposedphase_path=args.opposedphase_path,
        umap_path=args.umap_path,
        output_folder=args.output_folder,
        version=args.version,
        save_prediction=args.save_prediction,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()