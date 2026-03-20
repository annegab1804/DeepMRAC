import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import pandas as pd
from datetime import datetime
import nibabel as nib

def align_to_hu(volume: np.ndarray) -> np.ndarray:
    """
    Vérifie l'échelle du volume et le ramène en Unités Hounsfield (HU) si nécessaire.
    """
    # Si le minimum est autour de 0, les données ont probablement été décalées
    # (typiquement +1024 pour supprimer les valeurs négatives lors du deep learning)
    if volume.min() >= -50: 
        print(f"Alignement: Décalage détecté (min={volume.min():.2f}). Soustraction de 1024 pour repasser en HU.")
        # On crée une copie pour éviter de modifier l'objet original par référence
        return volume.copy() - 1024
    
    # Si le minimum est déjà autour de -1000 / -1024, c'est que c'est déjà en HU
    print(f"Alignement: Échelle HU correcte détectée (min={volume.min():.2f}).")
    return volume

def calculate_dice(
    image_a: np.ndarray,
    image_b: np.ndarray,
    threshold: float,
) -> float:
    """Calculates the Dice Similarity Coefficient (DSC) for a specific tissue class.

    This function binarizes the input arrays at a specific threshold to create masks
    before calculating overlap.

    Args:
        image_a: The first 3D volume (e.g., sCT).
        image_b: The second 3D volume (e.g., CT).
        threshold: The intensity value used to binarize the images. 

    Returns:
        The Dice coefficient as a float between 0.0 and 1.0.
    """
    mask_a = image_a > threshold
    mask_b = image_b > threshold
    
    intersection = np.logical_and(mask_a, mask_b).sum()
    return (2.0 * intersection) / (mask_a.sum() + mask_b.sum() + 1e-8)


def calculate_quality_metrics(
    sct_nii_path: str,
    ct_nii_path: str,
) -> dict[str, float]:
    """Computes metrics between a synthetic CT and the original CT.

    This function loads both DICOM series, ensures they have matching dimensions,
    and calculates standard image quality metrics to evaluate model performance.
    Computes bias estimates (RE, ARE, ME, MAE) plus PSNR, SSIM, and Dice.

    Args:
        sct_nii (str): Path to the NIftI image of the synthetic CT.
        ct_nii (str): Path to the  NIftI image of the original CT.

    Returns:
        A dictionary containing the calculated metrics:
            - "ME": Mean Error (lower is better).
            - "MAE": Mean Absolute Error (lower is better).
            - "RE": Relative Error (lower is better).
            - "ARE": Absolute Relative Error (lower is better).
            - "PSNR": Peak Signal-to-Noise Ratio (higher is better).
            - "SSIM": Structural Similarity Index Measure (closer to 1 is better).
            - "Dice": Dice Similarity Coefficient (closer to 1 is better).

    Raises:
        ValueError: If the shapes of the sCT and CT volumes do not match.
    """
    # Extract data
    sct_nii = nib.load(sct_nii_path)
    ct_nii = nib.load(ct_nii_path)

    sct_vol_raw = sct_nii.get_fdata().astype(np.float32)
    ct_vol_raw = ct_nii.get_fdata().astype(np.float32)

    # Ensure shapes match (Crucial for voxel-wise comparison)
    if sct_vol_raw.shape != ct_vol_raw.shape:
        raise ValueError(f"Shape mismatch: sCT {sct_vol_raw.shape} vs CT {ct_vol_raw.shape}")
    
    print("--- Vérification CT ---")
    ct_vol = align_to_hu(ct_vol_raw)
    
    print("--- Vérification sCT ---")
    sct_vol = align_to_hu(sct_vol_raw)
    
    print(f"sCT Range: {sct_vol.min():.2f} to {sct_vol.max():.2f}")
    print(f"CT Range: {ct_vol.min():.2f} to {ct_vol.max():.2f}")
    
    metrics_dict = {}

    modes = {
        'tissue': -500, # Tissue-air
        'bone': 300     # Bone-soft
    }
    
    for prefix, threshold in modes.items():
        mask = ct_vol > threshold
        sct_valid = sct_vol[mask]
        ct_valid = ct_vol[mask]
        
        if len(ct_valid) > 0:
            diff = sct_valid - ct_valid
            metrics_dict[f"{prefix}_ME"] = float(np.mean(diff))
            metrics_dict[f"{prefix}_MAE"] = float(np.mean(np.abs(diff)))
            metrics_dict[f"{prefix}_RE"] = float(np.mean(diff / np.abs(ct_valid)))
            metrics_dict[f"{prefix}_ARE"] = float(np.mean(np.abs(diff) / np.abs(ct_valid)))
            metrics_dict[f"{prefix}_Dice"] = float(calculate_dice(sct_vol, ct_vol, threshold))
        else:
            for m in ["ME", "MAE", "RE", "ARE", "Dice"]:
                metrics_dict[f"{prefix}_{m}"] = 0.0

    # Normalize for PSNR/SSIM (Metrics usually expect a defined range)
    # We use the max/min of the reference (CT)
    data_range = ct_vol.max() - ct_vol.min()
    metrics_dict["PSNR"] = float(peak_signal_noise_ratio(ct_vol, sct_vol, data_range=data_range))
    metrics_dict["SSIM"] = float(structural_similarity(ct_vol, sct_vol, data_range=data_range))

    return metrics_dict


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
        'Method': [rmi_type]
    }
    # Add all metrics (PSNR, SSIM, tissue_MAE, bone_MAE, etc.)
    for key, value in metrics_dict.items():
        new_data[key] = [value]

    df_new = pd.DataFrame(new_data)
    
    if os.path.isfile(csv_path):
        df_new.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        df_new.to_csv(csv_path, mode='w', index=False, header=True)
        
    print(f"Metrics saved at : {csv_path}")