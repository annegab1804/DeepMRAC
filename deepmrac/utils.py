import argparse, os, shutil
from pathlib import Path
import numpy as np 
import pydicom as dicom  
import nibabel as nib
import dicom2nifti
from nilearn.image import resample_img
from pydicom.uid import generate_uid

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sort_dicomfiles(
    source_folder: str,
    temp_subfolder: str,
    verbose: bool = False,
) -> None:
    """Sorts DICOM files by instance number into a structured temporary directory.

    Iterates through the patient folder, identifies DICOM files, and copies them 
    into a new 'dicom' subdirectory within the temp folder. Files are renamed 
    using their DICOM InstanceNumber (e.g., dicom001.ima) to ensure 
    numerical ordering.

    Args:
        source_folder (str): Path to the source directory containing DICOM files 
            (T1 weighted MPRAGE, Dixon, UTE and Umap).
        temp_subfolder (str): Path to the directory where the sorted 'dicom' 
            subfolder will be created.
        verbose (bool): If True, prints status messages when files are 
            discovered in a subdirectory.
            Defaults to False.

    Raises:
        FileExistsError: If the '{temp_folder}' directory already exists.
        AttributeError: If a file is encountered that lacks an 'InstanceNumber' 
            DICOM tag.
        RuntimeError: If two Dicom files have the same InstanceNumber?
    """
    if os.path.exists(temp_subfolder) and os.listdir(temp_subfolder):
        raise FileExistsError(f"Folder {temp_subfolder} is not empty !")
    
    os.makedirs(temp_subfolder, exist_ok=True)
        
    for root,subdirs,files in os.walk(source_folder):
        if len(subdirs) > 0 or not files:
            continue
        
        if verbose:
            print("Found files in %s. Making copy" % root)
        
        for f in files:
            if f.startswith('.'): 
                continue
            
            file_path = os.path.join(root, f)
            try:
                dcm = dicom.dcmread(file_path)
            except dicom.errors.InvalidDicomError:
                continue

            if "localizer" in getattr(dcm, "SeriesDescription", "").lower():
                continue

            if not hasattr(dcm, 'InstanceNumber'):
                raise AttributeError(f"File {f} lacks an 'InstanceNumber' DICOM tag.")

            dest_path = os.path.join(temp_subfolder, f"dicom{int(dcm.InstanceNumber)}.ima")
            if os.path.exists(dest_path):
                raise RuntimeError(f"InstanceNumber {dcm.InstanceNumber} already exists in {temp_subfolder}!")

            shutil.copyfile(file_path, dest_path)

def convert_interfile_to_nifti(hdr_path: str, output_nii_path: str) -> None:
    """Converts an HRRT Interfile volume to a standardized NIfTI image.

    This function parses the .i.hdr text file for dimensions and voxel sizes,
    reads the corresponding .i binary file, and reorders the data from the 
    Interfile (Z, Y, X) storage format to a standard NIfTI (X, Y, Z) structure.
    Finally, it ensures the output is saved in the canonical RAS orientation.

    Args:
        hdr_path (str): Path to the Interfile header (.i.hdr) file.
        output_nii_path (str): Full path where the .nii.gz file will be saved.

    Note:
        The function assumes the Interfile binary data starts at the 
        Left-Posterior-Inferior corner (LPS/RAS origin) by default.
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
    
    # Save
    nii_img = nib.Nifti1Image(volume_xyz, affine)
    nib.save(nii_img, output_nii_path)


def convert_dicom_to_nifti(
    dicom_dir: str,
    output_nii: str,
    verbose: bool = False,
) -> None:
    """Converts a DICOM series to a NIfTI image and validates the output.

    Reads the sorted DICOM files from the source directory and uses 
    dicom2nifti to generate a compressed NIfTI (.nii.gz) file. The resulting 
    image is automatically reoriented to the standard anatomical orientation.

    Args:
        dicom_dir (str): Path to the directory containing the DICOM series 
            (e.g., the 'dicom' subfolder with sorted .ima or .dcm files).
        output_nii (str): Full path (including filename) where the converted 
            .nii.gz file will be saved.
        verbose (bool): If True, prints status updates to the console during 
            the conversion process.
            Defaults to False.

    Raises:
        dicom2nifti.exceptions.ConversionError: If the DICOM series is 
            incomplete or inconsistent, preventing a valid conversion.
    """
    if verbose: 
        print(f'Converting {dicom_dir} to {output_nii}')

    dicom2nifti.dicom_series_to_nifti(
        dicom_dir, 
        output_nii, 
        reorient_nifti=True
        )
    
def load_and_resample_images(
    nii_image: str,
    verbose: bool = False,
) -> tuple[np.ndarray, nib.nifti1.Nifti1Image]:
    """Loads a NIfTI image and resamples it to a fixed isotropic resolution.

    The function loads the NIfTI file, calculates a new affine matrix to 
    achieve a 192^3 matrix size with 1.5626mm isotropic voxels, and centers 
    the volume.

    Note: This function does NOT perform the axis swaps/flips required by 
    the model; these must be applied to the returned numpy array externally.

    Args:
        nii_image (str): NIfTI image.
        verbose (bool): If True, prints status updates during the loading 
            and resampling process.
            Defaults to False.

    Returns:
        The resampled NIfTI image object with a 192^3 shape and 
        the calculated isotropic affine.
    """
    if verbose:
        print(f"Loading and resampling data from {nii_image}.")
        
    # Load dataset
    nii = nib.load(nii_image)

    # Resample to 192x192x192 and isotropic voxel size of 1.5626    
    target_shape = np.array((192,192,192))
    new_resolution = [1.5626,-1.5626,-1.5626]
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    nii_ref = resample_img(
        nii,
        target_affine=new_affine,
        target_shape=target_shape,
        interpolation='linear'
    )

    data = nii_ref.get_fdata()
    
    # Return nii handles as well as new image
    return data, nii_ref

def resample_to_output_format(
    pred_nii: nib.nifti1.Nifti1Image,
    umap_native: nib.nifti1.Nifti1Image,
    output_file: str,
    verbose: bool = False,
) -> np.ndarray:
    """Resample the predicted image back to the native Umap format.

    The function aligns the center of the predicted volume with the center of 
    the native volume in world coordinates to prevent spatial clipping. It 
    then performs linear interpolation to match the native matrix size and 
    voxel resolution.

    Args:
        pred_nii (nib.nifti1.Nifti1Image): The NIfTI predicted image (e.g., sCT).
        umap_native (nib.nifti1.Nifti1Image): The original native NIfTI image 
            defining the target geometry and coordinate system.
        output_file (str): Path where the resampled NIfTI file will be saved.
        verbose (bool): If True, prints status and affine matrices to console. 
            Defaults to False.

    Returns:
        np.ndarray: The resampled data array in the native coordinate system.
    """
    if verbose: print(f"Resampling to Umap format.")

    # Ensure both images are in the same canonical orientation (RAS)
    pred_nii = nib.as_closest_canonical(pred_nii)
    umap_native = nib.as_closest_canonical(umap_native)

    # Calculate world coordinates of the center of both volumes
    # Formula: Affine @ [center_voxel_coords, 1]
    target_center = umap_native.affine @ np.append(np.array(umap_native.shape[:3]) / 2.0, 1)
    pred_center = pred_nii.affine @ np.append(np.array(pred_nii.shape[:3]) / 2.0, 1)
    
    # Adjust the translation (4th column) of the prediction affine to match target center
    # This prevents "black images" caused by coordinate misalignment
    new_pred_affine = pred_nii.affine.copy()
    new_pred_affine[:3, 3] += (target_center[:3] - pred_center[:3])
    
    # Create a temporary NIfTI object with the corrected spatial position
    pred_nii_aligned = nib.Nifti1Image(pred_nii.get_fdata(), new_pred_affine)

    # Get background value for padding (usually the minimum intensity)
    data_src = pred_nii.get_fdata()
    fill_val = float(np.min(data_src))

    # Resampling
    pred_rsl = resample_img(
        pred_nii_aligned,
        target_affine=umap_native.affine, 
        target_shape=umap_native.shape,
        interpolation='linear',
    )

    # Ensure result is canonical and extract data
    pred_rsl = nib.as_closest_canonical(pred_rsl)
    data_rsl = pred_rsl.get_fdata()

    # Saving
    if verbose:
        print(f"Saving QC nii file to {output_file}.")
    nib.save(pred_rsl, output_file)
    
    return data_rsl

def to_dcm(
    DeepX: np.ndarray,
    dcmcontainer: Path,
    dicomfolder: str,
    rmi_type: str,
) -> None:
    """Overwrites DICOM templates with predicted pixel data and unique UIDs.

    Iterates through a template DICOM folder and replaces pixel data with 
    slices from DeepX. Updates SeriesInstanceUID and SOPInstanceUID.

    Note: 
        DeepX must be structured such that the first dimension (axis 0) 
        corresponds to the DICOM InstanceNumber index.

    Args:
        DeepX (np.ndarray): The 3D predicted image array (Expected shape: RAS).
        dcmcontainer (Path): Path to the folder containing template DICOM files.
        dicomfolder (str): Destination path where the new DICOM series will be saved.
        rmi_type (str): The name of the method of RMI used (eg.T1, UTE, Dixon or CT).

    Raises:
        FileNotFoundError: If the dcmcontainer folder is empty or does not exist.
        ValueError: If the number of slices in DeepX does not match the number 
            of DICOM files in dcmcontainer.
        KeyError: If rmi_type is not one of the expected keys ('T1', 'UTE', 'Dixon', 'CT').
    """
    def listdir_nohidden(path):
        return sorted([f for f in os.listdir(path) if not f.startswith('.')])
    
    # Read first file to get header information
    files = listdir_nohidden(dcmcontainer)

    if not files:
        raise FileNotFoundError(f"Template folder {dcmcontainer} is empty !")
    
    ds_template = dicom.dcmread(os.path.join(dcmcontainer, files[0]))
    
    # ImageOrientationPatient: [cos_r_x, cos_r_y, cos_r_z, cos_c_x, cos_c_y, cos_c_z]
    iop = ds_template.ImageOrientationPatient
    row_vec = np.array(iop[3:])
    col_vec = np.array(iop[:3])
    slice_vec = np.cross(col_vec, row_vec)

    # Determines which NumPy axis (0=X, 1=Y, 2=Z) correspond to which one in DICOM
    main_axis_row = np.argmax(np.abs(row_vec))
    main_axis_col = np.argmax(np.abs(col_vec))
    main_axis_slice = np.argmax(np.abs(slice_vec))

    DeepX = np.flip(DeepX, axis=0) # because RAS = [-LPS_x, -LPS_y, -LPS_z]
    DeepX = np.flip(DeepX, axis=1) # because RAS = [-LPS_x, -LPS_y, -LPS_z]
    np_DeepX = np.transpose(DeepX, (main_axis_slice, main_axis_row, main_axis_col))

    if slice_vec[main_axis_slice] < 0:
        np_DeepX = np.flip(np_DeepX, axis=0)

    # Flip the Rows
    if row_vec[main_axis_row] < 0: 
        np_DeepX = np.flip(np_DeepX, axis=1)

    # Flip the Cols
    if col_vec[main_axis_col] < 0:
        np_DeepX = np.flip(np_DeepX, axis=2)

    num_files = len(files)
    num_slices = np_DeepX.shape[0]
    if num_slices != num_files:
        raise ValueError(
            f"Dimension Mismatch: The template folder contains {num_files} DICOM files, "
            f"but DeepX has {num_slices} slices on axis 0. They must be equal."
        )
    
    pixel_type = ds_template.pixel_array.dtype
    np_DeepX = np.array(np_DeepX,dtype=pixel_type)
    largest_pixel_value = int(np_DeepX.max())

    # Generate unique SeriesInstanceUID
    new_series_uid = generate_uid()

    if not os.path.exists(dicomfolder):
        os.makedirs(dicomfolder, exist_ok=True)

    # Read each file in UMAP container, replace relevant tags
    for f in files:
        ds = dicom.dcmread(os.path.join(dcmcontainer, f))
        
        # Determine the slice index based on the original DICOM InstanceNumber
        i = int(ds.InstanceNumber) - 1
        
        # Skip if the instance number exceeds the available predicted slices
        if i < 0 or i >= np_DeepX.shape[0]:
            continue

        # Extract the corresponding 2D slice (Y-axis selection)
        slice_data = np_DeepX[i, :, :]

        ds.Rows, ds.Columns = slice_data.shape
        ds.LargestImagePixelValue = largest_pixel_value
        ds.PixelData = slice_data.tobytes() 

        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = f"Deep{rmi_type}_Predicted"
        series_nb_dict = {'T1': '507', 'UTE': '505', 'Dixon': '506', 'CT': '508'}
        ds.SeriesNumber = series_nb_dict[rmi_type]

        # Generate unique SOPInstanceUID
        ds.SOPInstanceUID = generate_uid()

        # Save the file
        output_fname = f"dicom_{int(ds.InstanceNumber):04d}.dcm"
        ds.save_as(os.path.join(dicomfolder, output_fname))

def to_interfile(
    DeepX: np.ndarray,
    hdr_template: Path,
    output_path: str,
    rmi_type: str,
    verbose: bool = False,
) -> None:
    """Creates a new Interfile volume by overwriting a template with predicted data.

    Args:
        DeepX (np.ndarray): Predicted image array in RAS order.
        hdr_template (Path): Path to the original .i.hdr file to use as a template.
        output_path (str): Directory where the new .i.hdr and .i files will be saved.
        rmi_type (str): Name of the method used (e.g., 'T1', 'CT').
        verbose (bool): If True, prints status messages to the console. 
            Defaults to False.

    Raises:
        FileNotFoundError: If the template header is not found.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Prepare file names
    base_name = f"Deep{rmi_type}_Predicted"
    new_hdr_path = os.path.join(output_path, f"{base_name}.i.hdr")
    new_bin_path = os.path.join(output_path, f"{base_name}.i")

    # Update Header Information
    new_header_lines = []
    with open(hdr_template, 'r') as f:
        for line in f:
            # Update the reference to the binary data file
            if "name of data file" in line.lower():
                new_header_lines.append(f"name of data file := {base_name}.i\n")
            # Update descriptions if needed
            elif "originating system" in line.lower():
                new_header_lines.append(f"originating system := DeepMRAC_{rmi_type}\n")
            else:
                new_header_lines.append(line)

    # Write the new header
    with open(new_hdr_path, 'w') as f:
        f.writelines(new_header_lines)

    # Write Binary Data
    # IMPORTANT: Ensure DeepX is back in the Interfile storage order (Z, Y, X)
    # and use the correct float32 type for HRRT.
    bin_data = DeepX.astype(np.float32)
    bin_data = np.transpose(bin_data, (2, 1, 0))
    bin_data = bin_data[::-1, ::-1, :]
    
    # Flatten the array to write it as a continuous binary stream
    bin_data.tofile(new_bin_path)

    if verbose:
        print(f" Interfile volume created:")
        print(f"   Header: {new_hdr_path}")
        print(f"   Binary: {new_bin_path}")
        print(f"   Final Shape: {DeepX.shape}")