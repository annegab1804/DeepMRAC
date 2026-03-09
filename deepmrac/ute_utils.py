import os, shutil, datetime
import numpy as np
import pydicom as dicom  

def sort_files(source_folder: str, temp_subfolder: str,  verbose:bool) -> None:
    """Sorts DICOM files into subdirectories named by SeriesNumber.

    Iterates through the source directory, identifies DICOM files, and organizes 
    them into a hierarchical structure within the temporary folder. Each 
    file is renamed using its InstanceNumber to ensure correct sequential ordering.

    The resulting structure is: {temp_subfolder}/dicom{InstanceNumber}.ima

    Args:
        temp_subfolder (str): Path to the directory containing raw DICOM files 
            (e.g., UTE, Umap).
        temp_folder (str): Path to the root directory where the sorted 
            series subfolders will be created.
        verbose (bool): If True, prints status messages when a directory 
            is being processed.

    Raises:
        AttributeError: If a DICOM file is missing the 'InstanceNumber' tag.
        FileNotFoundError: If the source_folder does not exist.
    """
    os.makedirs(f'{temp_subfolder}', exist_ok=True)
    
    for root,subdirs,files in os.walk(source_folder):
        
        if len(subdirs) > 0:
            continue
        if not len(files) > 0:
            continue
        
        if verbose:
            print(f"Processing files in: {root}")
        
        for f in files:
            if f.startswith('.'):
                continue
            dcm = dicom.dcmread(f"{root}/{f}")

            shutil.copyfile(
                os.path.join(root, f), 
                f"{temp_subfolder}/dicom{int(dcm.InstanceNumber)}.ima"
            )


def load_data(temp_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """Validates and loads UTE TE1 and TE2 DICOM series into numpy arrays.

    This function sorts subdirectories in the temporary folder, ensures that 
    the first three series each contain exactly 192 DICOM files, and loads 
    the pixel data from the first two series into 3D volumes.

    Args:
        temp_folder (str): Path to the directory containing the sorted 
            DICOM series subfolders (e.g., '101', '102').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 3D numpy arrays 
            (ute1, ute2) of shape (192, 192, 192).

    Raises:
        AssertionError: If any of the first three series directories do not 
            contain exactly 192 files.
        ValueError: If a folder name cannot be converted to float during sorting.
        AttributeError: If a DICOM file is missing the 'InstanceNumber' tag.
    """
    folders_to_check = [f for f in os.listdir(temp_folder) 
                    if (f.startswith('ute') or f.startswith('umap')) 
                    and os.path.isdir(os.path.join(temp_folder, f))]
    
    # Check that correct number of files is present
    for folder in folders_to_check:
        path_to_check = os.path.join(temp_folder, folder)
        assert len(os.listdir(path_to_check)) == 192, f"Error: {path_to_check} doesn't contain 192 files."
    
    # Load UTE TE1
    ute1 = np.empty((192,192,192))
    for filename in os.listdir(f"{temp_folder}/ute1_dcm"):
        ds = dicom.dcmread(f"{temp_folder}/ute1_dcm/{filename}")
        i = int(ds.InstanceNumber)-1
        ute1[i,:,:] = ds.pixel_array
        
    # Load UTE TE2
    ute2 = np.empty((192,192,192))
    for filename in os.listdir(f"{temp_folder}/ute2_dcm"):
        ds = dicom.dcmread(f"{temp_folder}/ute2_dcm/{filename}")
        i = int(ds.InstanceNumber)-1
        ute2[i,:,:] = ds.pixel_array
        
    return ute1,ute2
    
def to_dcm(DeepX: np.ndarray, dcmcontainer: str, dicomfolder: str) -> None:  
    """Overwrites DICOM templates with predicted pixel data and unique UIDs.

    This function iterates through a template DICOM folder, replaces the pixel 
    data with slices from the predicted array, and updates the UIDs (Series 
    and SOP) to ensure the new series is unique and importable into a PACS. 
    It also adjusts the orientation to correct for NIfTI-to-DICOM conversion.

    Args:
        DeepX (np.ndarray): The 3D predicted image array.
        dcmcontainer (Path): Path to the folder containing template DICOM files.
        dicomfolder (str): Destination path where the new DICOM series will be saved.

    Returns:
        None
    """
    
    def listdir_nohidden(path):
        return [f for f in os.listdir(path) if not f.startswith('.')]
    
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
    for f in listdir_nohidden(dcmcontainer):
        ds=dicom.dcmread(os.path.join(dcmcontainer,f))
        i = int(ds.InstanceNumber)-1
        
        ds.LargestImagePixelValue = int(largest_pixel_value)
        ds.PixelData = np_DeepX[i,:,:].tobytes() # Inserts actual image info

        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = "DeepUTE"
        ds.SeriesNumber = "505"

        # Generate unique SOPInstanceUID
        ds.SOPInstanceUID = f"{new_series_uid}.{i+1}"

        # Save the file
        output_fname = f"dicom_{int(ds.InstanceNumber):04d}.dcm"
        ds.save_as(os.path.join(dicomfolder, output_fname))
    