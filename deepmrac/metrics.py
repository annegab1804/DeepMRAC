import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import pandas as pd
from datetime import datetime
import nibabel as nib

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
    smu_nii_path: str,
    mu_nii_path: str,
) -> dict[str, float]:
    """Computes metrics between a synthetic umap and the original umap.

    This function loads both NIftI files, ensures they have matching dimensions,
    and calculates standard image quality metrics to evaluate model performance.
    Computes bias estimates (RE, ARE, ME, MAE) plus PSNR, SSIM, and Dice.

    Args:
        sct_nii (str): Path to the NIftI image of the synthetic umap.
        ct_nii (str): Path to the  NIftI image of the original umap.

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
        ValueError: If the shapes of the synthetic umap and umap volumes do not match.
    """
    # Extract data
    smu_nii = nib.load(smu_nii_path)
    mu_nii = nib.load(mu_nii_path)

    smu_vol = smu_nii.get_fdata().astype(np.float32)
    mu_vol = mu_nii.get_fdata().astype(np.float32)

    # Ensure shapes match (Crucial for voxel-wise comparison)
    if smu_vol.shape != mu_vol.shape:
        raise ValueError(f"Shape mismatch: synthetic umap {smu_vol.shape} vs umap {mu_vol.shape}")
    
    
    print(f"sUmap Range: {smu_vol.min():.2f} to {smu_vol.max():.2f}")
    print(f"Umap Range: {mu_vol.min():.2f} to {mu_vol.max():.2f}")
    
    metrics_dict = {}

    modes = {
        'tissue': 0.096, # Tissue-air 9.6×10-5×(0+1000)=0.096
        'bone': 0.15    # Bone-soft 5×10-5(1000+1000)+5×10-2=0.15
    }
    
    for prefix, threshold in modes.items():
        mask = mu_vol > threshold
        smu_valid = smu_vol[mask]
        mu_valid = mu_vol[mask]
        
        if len(mu_valid) > 0:
            diff = smu_valid - mu_valid
            metrics_dict[f"{prefix}_ME"] = float(np.mean(diff))
            metrics_dict[f"{prefix}_MAE"] = float(np.mean(np.abs(diff)))

            eps = 1e-6
            metrics_dict[f"{prefix}_RE"] = float(np.mean(diff / np.abs(mu_valid) + eps))
            metrics_dict[f"{prefix}_ARE"] = float(np.mean(np.abs(diff) / np.abs(mu_valid) + eps))
            
            metrics_dict[f"{prefix}_Dice"] = float(calculate_dice(smu_vol, mu_vol, threshold))
        else:
            for m in ["ME", "MAE", "RE", "ARE", "Dice"]:
                metrics_dict[f"{prefix}_{m}"] = 0.0

    # Normalize for PSNR/SSIM (Metrics usually expect a defined range)
    # We use the max/min of the reference (Umap)
    data_range = mu_vol.max() - mu_vol.min()
    metrics_dict["PSNR"] = float(peak_signal_noise_ratio(mu_vol, smu_vol, data_range=data_range))
    metrics_dict["SSIM"] = float(structural_similarity(mu_vol, smu_vol, data_range=data_range))

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

def calculate_synthesis_metrics(
    tabs_list: list[pd.DataFrame]
) -> list:
    """Calculate Mean and std for different metric by methods.

    Agrs: 
        tabs_list (list[pd.DataFrame]): List of pandas dataframe with a column Method.

    Returns:
        A dataframe with the number of patients and the mean +- std of each metric
            grouped by the Method column.
    """
    if not tabs_list:
        return None
    
    whole_df = pd.concat(tabs_list, ignore_index=True)
    metrics = whole_df.select_dtypes(include=['number']).columns.tolist()
    
    stats = whole_df.groupby('Method')[metrics].agg(['mean', 'std', 'count'])

    df_final = pd.DataFrame(index=stats.index)
    df_final['N'] = stats[metrics[0]]['count'].astype(int)

    for m in metrics:
        mean_vals = stats[m]['mean']
        std_vals = stats[m]['std']
        df_final[m] = mean_vals.map('{:.3f}'.format) + " ± " + std_vals.map('{:.3f}'.format)

    return df_final