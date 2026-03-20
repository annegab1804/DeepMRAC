import argparse
import tempfile
import shutil
import os
import nibabel as nib
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
from deepmrac.predictions import predict_DeepDixon
from deepmrac.metrics import calculate_quality_metrics, save_metrics_to_csv
from deepmrac.plots import plot_comparison


def run_pipeline(
    inphase_path: str,
    opposedphase_path: str,
    umap_path: str,
    output_folder: str,
    ct_path: str | None = None,
    version: str = 'VE11P',
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
        umap_path: Path to the directory containing Umap (template) DICOM or interfile files.
        output_folder: Path where the resulting MRAC DICOM or interfile files will be saved.
        ct_path: Path to folder with dicom files or path to the nifti file of original CT.. 
            Defaults to None.
        version: Model training version to use (e.g., 'VB20P' or 'VE11P'). 
            Defaults to 'VE11P'.
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
        # Sort and convert files into specific folders
        # --- Process Dixon (Standard DICOM) ---
        sort_dicomfiles(source_folder=inphase_path, temp_subfolder=f"{tmpdir}/inphase_dcm", verbose=verbose )
        sort_dicomfiles(source_folder=opposedphase_path, temp_subfolder=f"{tmpdir}/opposedphase_dcm", verbose=verbose )
        convert_dicom_to_nifti(dicom_dir=f"{tmpdir}/inphase_dcm", output_nii=f"{tmpdir}/inphase.nii.gz", verbose=verbose)
        convert_dicom_to_nifti(dicom_dir=f"{tmpdir}/opposedphase_dcm", output_nii=f"{tmpdir}/opposedphase.nii.gz", verbose=verbose)

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
        
        # Load and resample data
        inphase_rsl, inphase_ref = load_and_resample_images(nii_image=f"{tmpdir}/inphase.nii.gz", verbose=verbose)
        opposedphase_rsl, opposedphase_ref = load_and_resample_images(nii_image=f"{tmpdir}/opposedphase.nii.gz", verbose=verbose)
        umap_nat = nib.load(f'{tmpdir}/umap.nii.gz')
    
        # Predict
        pred = predict_DeepDixon(inphase=inphase_rsl, opposedphase=opposedphase_rsl, version=version)

        # Resample to Umap Formatos.makedirs(output_folder, exist_ok=True)
        pred_nii = nib.Nifti1Image(pred, inphase_ref.affine, inphase_ref.header)
        os.makedirs(output_folder, exist_ok=True)
        DeepX = resample_to_output_format(pred_nii=pred_nii, umap_native=umap_nat, verbose=verbose, output_file=f"{output_folder}/DeepDixon.nii.gz")
       
        # Final DICOM (using Umap as container)
        if interfile_headers:
            to_interfile(DeepX=DeepX, hdr_template=hdr_full_path, output_path=f"{output_folder}/DeepDixon", rmi_type="Dixon")
        else:
            to_dcm(DeepX=DeepX, dcmcontainer=f"{tmpdir}/umap_dcm", dicomfolder=f"{output_folder}/DeepDixon", rmi_type="Dixon")
                
        print(f"Success! Result saved in: {output_folder}")

        ct_nii_path = None

        if ct_path and os.path.exists(ct_path):
            is_nifti = ct_path.lower().endswith(('.nii', '.nii.gz'))

            if is_nifti:
                if verbose: print(f"Input is already NIfTI: {ct_path}")
                ct_nii_path = ct_path
            else:
                # DICOM files
                sort_dicomfiles(source_folder=ct_path, temp_subfolder=f"{tmpdir}/ct_dcm", verbose=verbose)
                convert_dicom_to_nifti(dicom_dir=f"{tmpdir}/ct_dcm", output_nii=f"{tmpdir}/ct.nii.gz", verbose=verbose)
                ct_nii_path = f'{tmpdir}/ct.nii.gz'

            # Resample using Umap's grid size
            ct_nat = nib.load(ct_nii_path)
            ct_rsl = resample_to_output_format(pred_nii=ct_nat, umap_native=umap_nat, verbose=verbose, output_file=f"{output_folder}/CT_resampled.nii.gz")

            # Calculate metrics
            metrics_dict = calculate_quality_metrics(
                sct_nii_path=f"{output_folder}/DeepDixon.nii.gz",
                ct_nii_path=f"{output_folder}/CT_resampled.nii.gz",
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

            save_metrics_to_csv(metrics_dict=metrics_dict, rmi_type='Dixon', output_folder=f"{output_folder}/metrics")

        # Plot
        plot_comparison(input_path=f"{tmpdir}/inphase.nii.gz", prediction_path=f"{output_folder}/DeepDixon.nii.gz", umap_path=f'{tmpdir}/umap.nii.gz', sct_path=ct_nii_path, model_type="Dixon", output_path=f"{output_folder}/comparison_plot.png")

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
    parser = argparse.ArgumentParser(description='Predict using DeepDixon.')
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
        help="Path to folder with dicom or interphile files of Umap.", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--ct_path", 
        help="Path to folder with dicom files or Path to the nifti file of original CT.", 
        type=str,
        default=None,
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
        default='VE11P',
        required=False,
    )
    parser.add_argument(
        "--verbose", 
        type=str2bool, 
        default=False,
        required=False,
    )
    args = parser.parse_args()

    run_pipeline(
        inphase_path=args.inphase_path,
        opposedphase_path=args.opposedphase_path,
        umap_path=args.umap_path,
        ct_path=args.ct_path,
        output_folder=args.output_folder,
        version=args.version,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()