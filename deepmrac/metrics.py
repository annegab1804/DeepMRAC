import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import pandas as pd
from datetime import datetime

def load_dicom_series(
    directory: str
) -> np.ndarray:
    """Loads a directory of DICOM or IMA files into a 3D numpy array.

    Uses SimpleITK to read a series of slices from a directory, ensuring correct
    spatial ordering based on DICOM metadata.

    Args:
        directory: Path to the folder containing .dcm or .IMA files.

    Returns:
        A 3D numpy array of type float32 containing the image volume.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image).astype(np.float32)

def calculate_dice(
    image_a: np.ndarray,
    image_b: np.ndarray,
    threshold: float = -500
) -> float:
    """Calculates the Dice Similarity Coefficient (DSC) for a specific tissue class.

    Since medical volumes are continuous (Hounsfield Units), this function 
    binarizes the input arrays at a specific threshold to create masks before 
    calculating overlap.

    Args:
        image_a: The first 3D volume (e.g., sCT).
        image_b: The second 3D volume (e.g., Umap).
        threshold: The intensity value used to binarize the images. 
            Defaults to -500 (standard for body/air separation).

    Returns:
        The Dice coefficient as a float between 0.0 and 1.0.
    """
    mask_a = image_a > threshold
    mask_b = image_b > threshold
    
    intersection = np.logical_and(mask_a, mask_b).sum()
    return (2.0 * intersection) / (mask_a.sum() + mask_b.sum())

def calculate_quality_metrics(
    sct_path: str,
    umap_path: str,
    dice_threshold: float = -500
) -> dict[str, float]:
    """Computes MAE, PSNR, SSIM, and Dice metrics between a synthetic CT and a Umap.

    This function loads both DICOM series, ensures they have matching dimensions,
    and calculates standard image quality metrics to evaluate model performance.
    MAE is particularly useful for quantifying error in Hounsfield Units (HU).

    Args:
        sct_path: Path to the folder containing the generated sCT DICOM files.
        umap_path: Path to the folder containing the reference Umap IMA/DICOM files.
        dice_threshold: The Hounsfield Unit threshold for the Dice calculation.
            Defaults to -500.

    Returns:
        A dictionary containing the calculated metrics:
            - "MAE": Mean Absolute Error (lower is better).
            - "PSNR": Peak Signal-to-Noise Ratio (higher is better).
            - "SSIM": Structural Similarity Index Measure (closer to 1 is better).
            - "Dice": Dice Similarity Coefficient (closer to 1 is better).

    Raises:
        ValueError: If the shapes of the sCT and Umap volumes do not match.
    """
    # Load volumes
    sct_vol = load_dicom_series(sct_path)
    umap_vol = load_dicom_series(umap_path)

    # Ensure shapes match (Crucial for voxel-wise comparison)
    if sct_vol.shape != umap_vol.shape:
        raise ValueError(f"Shape mismatch: sCT {sct_vol.shape} vs Umap {umap_vol.shape}")
    
    # Calculate Mean Absolute Error (MAE)
    mae_val = np.mean(np.abs(sct_vol - umap_vol))

    # Normalize for PSNR/SSIM (Metrics usually expect a defined range)
    # We use the max/min of the reference (Umap)
    data_range = umap_vol.max() - umap_vol.min()

    # Calculate PSNR
    psnr_val = peak_signal_noise_ratio(umap_vol, sct_vol, data_range=data_range)

    # Calculate SSIM
    ssim_val = structural_similarity(umap_vol, sct_vol, data_range=data_range)

    # Calculate Dice (Thresholded for 'Non-Air' pixels)
    dice_val = calculate_dice(sct_vol, umap_vol, threshold=dice_threshold)

    return {
        "MAE": float(mae_val),
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "Dice": dice_val,
    }

def save_metrics_to_csv(
    metrics_dict: dict,
    rmi_type: str,
    output_folder: str,
    filename: str = "all_metrics.csv"
) -> None :
    """Save metrics in a csv file.

    If the file already exists, the new metrics are added to it.
    
    Args:
        metrics_dict (dict): Dictionary containing MAE, PSNR, SSIM, Dice.
        rmi_type (str): The name of the method of RMI used (eg.T1, UTE or
            Dixon).
        output_folder (str): Folder where the csv will be saved.
        filename (str): Name of the CSV file. Default to 
    """
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, filename)
    
    new_data = {
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Method': [rmi_type],
        'MAE': [metrics_dict.get('MAE')],
        'PSNR': [metrics_dict.get('PSNR')],
        'SSIM': [metrics_dict.get('SSIM')],
        'Dice': [metrics_dict.get('Dice')]
    }
    
    df_new = pd.DataFrame(new_data)
    
    if os.path.isfile(csv_path):
        df_new.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        df_new.to_csv(csv_path, mode='w', index=False, header=True)
        
    print(f"Metrics saved at : {csv_path}")