import os, shutil, datetime
from pathlib import Path
import numpy as np 
import pydicom as dicom  
import nibabel as nib
import dicom2nifti
from nilearn.image import resample_img

def sort_files(
    source_folder: str,
    temp_subfolder: str,
    verbose: bool
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

    Raises:
        FileExistsError: If the '{temp_folder}' directory already exists.
        AttributeError: If a file is encountered that lacks an 'InstanceNumber' 
            DICOM tag.
    """
    os.makedirs(f'{temp_subfolder}', exist_ok=True)
    
    for root,subdirs,files in os.walk(source_folder):
        
        if len(subdirs) > 0:
            continue
        if not len(files) > 0:
            continue
        
        if verbose:
            print("Found files in %s. Making copy" % root)
        
        for f in files:
            if f.startswith('.'):
                continue
            dcm = dicom.dcmread(f"{root}/{f}")

            shutil.copyfile(
                os.path.join(root, f), 
                f"{temp_subfolder}/dicom{int(dcm.InstanceNumber)}.ima"
            )

def convert_to_nifti(
    dicom_dir: str,
    output_nii: str,
    verbose: bool
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
    verbose: bool
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
    pred: np.ndarray,
    model_ref: nib.nifti1.Nifti1Image,
    umap_native: nib.nifti1.Nifti1Image,
    rmi_type: str,
    verbose: bool = True,
    save_prediction: bool = False,
) -> np.ndarray:
    """Resample the predicted image back to umap format.

    The function creates a NIfTI object using the reference affine, and
    performs a linear interpolation to match the native matrix size and
    voxel resolution of the patient data.

    Important: 'pred' must be re-oriented (flips/swaps reversed) to match 
    the model_ref.affine orientation BEFORE calling this function.

    Args:
        pred (np.ndarray): The 3D predicted image array (192, 192, 192).
        model_ref (nib.nifti1.Nifti1Image): The NIfTI object used during 
            the prediction stage.
        umap_native (nib.nifti1.Nifti1Image): The original DICOM-derived 
            NIfTI image (Umap) defining the target geometry.
        rmi_type (str): The name of the method of RMI used (eg.T1, UTE or
            Dixon).
        verbose (bool): If True, prints status messages to the console. 
            Defaults to False.
        save_prediction (bool): If True, saves the resampled volume as 
            'DeepT1_QC.nii.gz' for quality control. Defaults to False.

    Returns:
        np.ndarray: The resampled prediction data in the native coordinate 
            system and resolution.
    """
    if verbose: print('Resampling to Umap format')
    
    pred_nii = nib.Nifti1Image(pred, model_ref.affine, model_ref.header)
    
    # Resample using Umap's grid size
    pred_rsl = resample_img(
        pred_nii,
        target_affine=umap_native.affine, 
        target_shape=umap_native.shape,  # Fit Umap matrix
        interpolation='linear'
    )

    # Save intermediate nii file
    if save_prediction:
        if verbose:
            print(f"Saving QC nii file to Deep{rmi_type}_QC.nii.gz")
        nib.save(pred_rsl,f'Deep{rmi_type}_QC.nii.gz')
    
    return pred_rsl.get_fdata()

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
        DeepX (np.ndarray): The 3D predicted image array.
        dcmcontainer (Path): Path to the folder containing template DICOM files.
        dicomfolder (str): Destination path where the new DICOM series will be saved.
        rmi_type (str): The name of the method of RMI used (eg.T1, UTE or
            Dixon).
    """
    def listdir_nohidden(path):
        return sorted([f for f in os.listdir(path) if not f.startswith('.')])
    
    # Read first file to get header information
    files = listdir_nohidden(dcmcontainer)
    if not files:
        raise FileNotFoundError(f"Template folder {dcmcontainer} is empty !")

    ds_template = dicom.dcmread(os.path.join(dcmcontainer, files[0]))
    pixel_type = ds_template.pixel_array.dtype

    np_DeepX = np.array(DeepX,dtype=pixel_type)
    largest_pixel_value = int(np_DeepX.max())

    # Generate unique SeriesInstanceUID
    now = datetime.datetime.now()
    uid_stamp = now.strftime("%Y%m%d%H%M%S%f")
    new_series_uid = f"1.3.12.2.1107.5.2.38.51014.{uid_stamp}.11111.0.0.0"

    if not os.path.exists(dicomfolder):
        os.makedirs(dicomfolder, exist_ok=True)

    # Read each file in UMAP container, replace relevant tags
    for f in files:
        ds = dicom.dcmread(os.path.join(dcmcontainer, f))
        i = int(ds.InstanceNumber) - 1
        
        if i >= np_DeepX.shape[2]:
            continue

        slice_data = np_DeepX[i, :, :]
        ds.Rows, ds.Columns = slice_data.shape
        ds.LargestImagePixelValue = largest_pixel_value
        ds.PixelData = slice_data.tobytes() 

        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = f"Deep{rmi_type}_Predicted"
        series_nb_dict = {'T1': '507', 'UTE': '505', 'Dixon': '506'}
        ds.SeriesNumber = series_nb_dict[rmi_type]

        # Generate unique SOPInstanceUID
        ds.SOPInstanceUID = f"{new_series_uid}.{i+1}"

        # Save the file
        output_fname = f"dicom_{int(ds.InstanceNumber):04d}.dcm"
        ds.save_as(os.path.join(dicomfolder, output_fname))