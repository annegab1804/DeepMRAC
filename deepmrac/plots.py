import matplotlib.pyplot as plt
import pydicom
import os
import nibabel as nib
import numpy as np
import dicom2nifti
import pydicom
import tempfile
from typing import Literal

def check_dicom_series(folder: str) -> None:
    """Checks and prints metadata for the first DICOM file in a series.

    This function scans a directory, reads the first file found, and displays
    key DICOM attributes such as Series Description, Modality, and Matrix size.

    Args:
        folder (str): The path to the directory containing DICOM files.

    Returns:
        None
    """
    all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    nb_files = len(all_files)
    
    if nb_files == 0:
        print(f"The folder {folder} is empty.")
        return

    first_file = all_files[0]
    ds = pydicom.dcmread(os.path.join(folder, first_file))
    
    print(f"Folder: {folder}")
    print(f"  - Description: {ds.SeriesDescription}")
    print(f"  - Modality:    {ds.Modality}")
    print(f"  - Matrix:     {ds.Rows}x{ds.Columns}")
    print(f"  - Files number: {nb_files}")
    if 'EchoTime' in ds:
        print(f"  - Echo Time: {ds.EchoTime} ms")
    field = getattr(ds, 'MagneticFieldStrength', None)
    if field:
        print(f"  - Field: {field}T")
    print("-" * 30)

def plot_dicom_file(file: str) -> None:
    """Reads a DICOM file and displays the image using matplotlib.

    Calculates and prints basic statistics (min, max, mean) of the pixel data 
    before rendering the grayscale image.

    Args:
        file (str): The file path to the DICOM image.

    Returns:
        None
    """
    ds = pydicom.dcmread(file)

    data = ds.pixel_array
    print(f"Max value: {data.max()}")
    print(f"Min value: {data.min()}")
    print(f"Mean value: {data.mean()}")
    print(f"Shape : {ds.Rows}x{ds.Columns}")
    
    plt.imshow(data, cmap='gray')
    plt.title(f"Slice {ds.InstanceNumber} - {ds.SeriesDescription}")
    plt.show()

def scan_dicom_folder_for_data(folder: str) -> None:
    """Scans a folder for valid DICOM files and validates pixel data presence.

    Iterates through all files ending in .ima or .dcm and checks if the 
    maximum pixel intensity is greater than zero to identify "valid" data.

    Args:
        folder (str): The path to the directory to scan.

    Returns:
        None
    """
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.ima', '.dcm'))])
    print(f"{len(files)} files found...")
    
    valid_files_count = 0
    
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            pixel_max = ds.pixel_array.max()
            pixel_min = ds.pixel_array.min()
            
            if pixel_max > 0:
                print(f"{os.path.basename(f)} | Max : {pixel_max} | Min: {pixel_min}")
                valid_files_count += 1
                
        except Exception as e:
            print(f"An error occured when trying to read {os.path.basename(f)} : {e}")
            
    print("-" * 30)
    if valid_files_count > 0:
        print(f"Result : {valid_files_count}/{len(files)} files have data.")
    else:
        print("Result : No data found in any of the files.")

def plot_3d_views(
    volume: np.ndarray,
    aspects: dict | None = None,
    rotation_map: dict | None = None,
    flip_map: dict | None = None,
) -> None:
    """Plots 3D orthogonal views from a pre-loaded NumPy volume.

    Args:
        volume (np.ndarray): 3D array of the imaging data (Z, Y, X order preferred).
        aspects (dict, optional): Pixel aspect ratios for each plane. 
            e.g., {'axial': 1.0, 'coronal': 2.5, 'sagittal': 2.5}. Defaults to None.
        rotation_map (dict, optional): Number of 90-degree CCW rotations per plane.
        flip_map (dict, optional): Axis to flip per plane (0 for vertical, 1 for horizontal).

    Returns:
        None: Displays a matplotlib figure.
    """
    if rotation_map is None: rotation_map = {}
    if flip_map is None: flip_map = {}
    
    # Default aspects to 1.0 if not provided
    if aspects is None:
        aspects = {'axial': 1.0, 'coronal': 1.0, 'sagittal': 1.0}

    # Extract center slices
    # Assumes volume is (Z, Y, X)
    z_mid, y_mid, x_mid = np.array(volume.shape) // 2
    
    views = {
        'axial': volume[z_mid, :, :],
        'coronal': volume[:, y_mid, :],
        'sagittal': volume[:, :, x_mid]
    }

    # Apply transformations and update aspect ratios
    planes = ['axial', 'coronal', 'sagittal']
    for plane in planes:
        # 1. Rotation
        k = rotation_map.get(plane, 0)
        if k != 0:
            views[plane] = np.rot90(views[plane], k=k)
            if k % 2 != 0:
                aspects[plane] = 1.0 / aspects[plane]
        
        # 2. Flip
        if plane in flip_map:
            views[plane] = np.flip(views[plane], axis=flip_map[plane])

    vmin, vmax = np.percentile(volume, [0.5, 99.5])

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, plane in enumerate(planes):    
        axes[i].imshow(
            views[plane], 
            cmap='gray', 
            aspect=aspects.get(plane, 1.0), 
            origin='lower',
            vmin=vmin,
            vmax=vmax
        )
        axes[i].set_title(f"{plane.capitalize()} view")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def extracts_nifti_volume_and_aspects(
    img_nii: nib.nifti1.Nifti1Image,
    verbose: bool = False
) -> tuple[np.ndarray, dict[str, float]]:
    """Standardizes a NIfTI image to RAS orientation and extracts volume/aspect ratios.

    Args:
        img_nii (nib.nifti1.Nifti1Image): The input NIfTI image object.
        verbose(bool, optional): If True print the spacing and the final volume shape.
            Defaults to False.

    Returns:
        tuple[np.ndarray, dict[str, float]]: A tuple containing:
            - volume (np.ndarray): The image data reshaped to (Z, Y, X).
            - aspects (dict): Ratios for 'axial', 'coronal', and 'sagittal' views.
    """
    # Force RAS+ orientation (Standard Neuro: Right-Anterior-Superior)
    # X=Left/Right, Y=Posterior/Anterior, Z=Inferior/Superior
    canonical_img = nib.as_closest_canonical(img_nii)
    
    # Get data and transpose from (X, Y, Z) to (Z, Y, X)
    volume = canonical_img.get_fdata().transpose(2, 1, 0)
    
    # Extract voxel spacing (mm) from the header
    dx, dy, dz = canonical_img.header.get_zooms()[:3]

    if verbose:
        print(f"Standardized Spacing (RAS): dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
        print(f"Final Volume Shape (ZYX): {volume.shape}")

    # Calculate aspect ratios for visualization (Matplotlib/Napari)
    aspects = {
        'axial': dy / dx,     # Y-X plane
        'coronal': dz / dx,   # Z-X plane
        'sagittal': dz / dy   # Z-Y plane
    }
    
    return volume, aspects

def load_nifti_volume_and_aspects(nifti_path: str) -> tuple[np.ndarray, dict[str, float]]:
    """Loads a NIfTI file from disk and extracts its volume and aspects.

    Args:
        nifti_path (str): Path to the .nii or .nii.gz file.

    Returns:
        tuple[np.ndarray, dict[str, float]]: Standardized volume (Z, Y, X) and aspect ratios.
    """
    img = nib.load(nifti_path)
    return extracts_nifti_volume_and_aspects(img)

def load_dicom_volume_and_aspects(folder_path: str)-> tuple[np.ndarray, dict[str, float]]:
    """Converts a DICOM series to NIfTI in memory and extracts volume and aspects.

    Args:
        folder_path (str): Path to the directory containing DICOM slices.

    Returns:
        tuple[np.ndarray, dict[str, float]]: Standardized volume (Z, Y, X) and aspect ratios.
    """
    # Create a temporary directory to handle the DICOM-to-NIfTI conversion safely
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, 'temp_result.nii.gz')
        
        # Convert DICOM series to NIfTI file
        # reorient_nifti=True ensures the output is in a standard orientation
        dicom2nifti.dicom_series_to_nifti(folder_path, output_file, reorient_nifti=True)
        
        # Load the newly created NIfTI file
        img = nib.load(output_file)
        
        # Pass to the extraction utility (which ensures RAS orientation)
        return extracts_nifti_volume_and_aspects(img)

def load_interfile_volume_and_aspects(hdr_path: str) -> tuple[np.ndarray, dict[str, float]]:
    """Parses HRRT Interfile header to extract volume and pixel aspect ratios.

    This function reads the .i.hdr text file to find matrix dimensions and 
    scaling factors, then reads the corresponding binary file (.i).

    Args:
        hdr_path (str): Path to the .i.hdr file.

    Returns:
        A tuple containing:
            - volume: The 3D numpy array reshaped to (Z, Y, X).
            - aspects: A dictionary with 'axial', 'coronal', and 'sagittal' ratios.

    Raises:
        ValueError: If the number of voxels read does not match the header dimensions.
    """
    # Read the header
    header = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            if ':=' in line:
                k, v = line.split(':=')
                header[k.strip().lower()] = v.strip()

    # matrix size [1]=X, [2]=Y, [3]=Z
    dim = [int(header.get(f'matrix size [{i}]', 0)) for i in [1, 2, 3]]
    vox_size = [float(header.get(f'scaling factor (mm/pixel) [{i}]', 1.0)) for i in [1, 2, 3]]

    # Load binary data
    img_path = hdr_path.replace('.i.hdr', '.i')
    data = np.fromfile(img_path, dtype=np.float32)
    
    # IMPORTANT: Reshape to (Z, Y, X) first because that's how Interfile stores it
    volume_zyx = data.reshape((dim[2], dim[1], dim[0]))
    volume_zyx = volume_zyx[::-1, ::-1, :]

    # Reorder to (X, Y, Z) for NIfTI standard
    # This is what allows 'as_closest_canonical' to work later
    volume_xyz = volume_zyx.transpose(2, 1, 0)

    # Create a standard affine (X, Y, Z)
    # We assume RAS orientation for HRRT by default
    affine = np.diag([vox_size[0], vox_size[1], vox_size[2], 1.0])
    nii_img = nib.Nifti1Image(volume_xyz, affine)
    return extracts_nifti_volume_and_aspects(nii_img)

def plot_comparison(
    input_path: str, 
    prediction_path: str, 
    umap_path: str,
    model_type: Literal['T1', 'UTE', 'Dixon'] = 'T1',
    sct_path: str | None  = None,
    output_path: str | None = None
) -> None:
    """Generates a comparison plot for DeepT1, DeepUTE, or DeepDixon models.

    Args:
        input_path (str): Path to the main input (T1, UTE Echo 1, or Dixon In-phase).
        prediction_path (str): Path to the DL model's prediction NIfTI file.
        umap_path (str): Path to the Umap template NIfTI file.
        model_type (str, optional): Type of model to adapt labels ('T1', 'UTE', or 'Dixon'). 
            Defaults to 'T1'.
        sct_path (str, optional): Optional path to a reference sCT.
            Defaults to None.
        output_path (str, optional): Optional path to save the resulting figure.
            Defaults to None.

    Returns:
        None. Displays and optionally saves the comparison grid.
    """
    # Mapping of model types to their specific input names
    input_names = {
        'T1': "T1-Weighted",
        'UTE': "UTE Echo 1",
        'Dixon': "Dixon In-phase"
    }
    
    model_name = f"Deep{model_type}"
    main_input_label = input_names.get(model_type, "Input MRI")

    # Build lists dynamically
    paths = [input_path, prediction_path, umap_path]
    names = [main_input_label, model_name, "Umap"]
    
    if sct_path:
        # Insert reference sCT before Umap for direct visual comparison
        paths.insert(2, sct_path)
        names.insert(2, "sCT Reference")
        
    planes = ['axial', 'coronal', 'sagittal']
    num_cols = len(paths)
    
    fig, axes = plt.subplots(len(planes), num_cols, figsize=(4 * num_cols, 12))
    
    # Handle single column case for matplotlib consistency
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (path, name) in enumerate(zip(paths, names)):
        volume, aspects = load_nifti_volume_and_aspects(path)
        
        # Central slices (Z, Y, X)
        z_mid, y_mid, x_mid = np.array(volume.shape) // 2
        
        views_data = {
            'axial': volume[z_mid, :, :],
            'coronal': volume[:, y_mid, :],
            'sagittal': volume[:, :, x_mid]
        }
        
        # Contrast normalization
        vmin, vmax = np.percentile(volume, [1, 99])

        for row_idx, plane in enumerate(planes):
            ax = axes[row_idx, col_idx]
            
            ax.imshow(
                views_data[plane], 
                cmap='gray', 
                vmin=vmin, 
                vmax=vmax, 
                aspect=aspects[plane], 
                origin='lower'
            )
            
            if row_idx == 0:
                ax.set_title(name, fontsize=14, fontweight='bold', pad=15)
            if col_idx == 0:
                ax.set_ylabel(plane.capitalize(), fontsize=14, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05) 

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {output_path}")

    plt.show()