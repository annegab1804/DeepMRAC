import numpy as np
import tensorflow as tf
from numpy.lib import stride_tricks 
from typing import Any


def cutup(
    data: np.ndarray,
    blck: tuple | np.ndarray,
    strd: tuple | np.ndarray
) -> np.ndarray:
    """Helper function to extract overlapping patches using NumPy stride tricks.

    This function calculates a new memory layout (strides) to create a multi-dimensional
    view of the input array, effectively 'cutting' it into patches without 
    duplicating data in memory.

    Args:
        data (np.ndarray): The input n-dimensional volume (e.g., 192x192x192).
        blck (tuple | np.ndarray): The dimensions of the patch/block (e.g., 16x192x192).
        strd (tuple | np.ndarray): The step size between patches for each dimension.

    Returns:
        np.ndarray: A high-dimensional view of the original array containing the patches.
    """
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6

def check_output(x: Any) ->  bool:
    """Verifies if an object is a valid array-like structure with a shape attribute.

    Args:
        x (Any): The object to validate (usually a NumPy array).

    Returns:
        bool: True if the object has a 'shape' attribute, False otherwise.
    """
    try:
        x.shape
        return True
    except:
        return False

def get_patches_znorm(
    vol1: np.ndarray, 
    vol2: np.ndarray | None = None, 
    normalize_both: bool = True
) -> np.ndarray:
    """Normalizes input volumes and extracts 16-slice deep patches.

    Applies Z-score normalization based only on non-zero voxels (background exclusion).
    If two volumes are provided, they can be normalized independently or using 
    the statistics of the second volume (useful for Dixon sequences).

    Args:
        vol1 (np.ndarray): Primary input volume.
        vol2 (np.ndarray, optional): Secondary input volume (e.g., Opposed-phase).
        normalize_both (bool): If True, each volume is normalized by its own mean/std.
            If False, vol1 is normalized using vol2's statistics. Defaults to True.

    Returns:
        np.ndarray: A float32 array of patches with shape (N, 16, 192, 192, Channels).
    """
    mean_vol1, std_vol1 = ( np.mean(vol1[np.where(vol1>0)]), np.std(vol1[np.where(vol1>0)]) )
    
    if check_output(vol2):
        mean_vol2, std_vol2 = ( np.mean(vol2[np.where(vol2>0)]), np.std(vol2[np.where(vol2>0)]) )
    
    # STANDARDIZE
    vol1 = np.true_divide( vol1 - mean_vol1, std_vol1 ) if normalize_both else np.true_divide( vol1 - mean_vol2, std_vol2 )
    if check_output(vol2):
        vol2 = np.true_divide( vol2 - mean_vol2, std_vol2 )

    patches_vol1 = cutup(vol1,(16,192,192),(2,1,1))
    if check_output(vol2):
        patches_vol2 = cutup(vol2,(16,192,192),(2,1,1))

    ijk = patches_vol1.shape[0]*patches_vol1.shape[1]*patches_vol1.shape[2]
    
    if check_output(vol2):
        selected_patches = np.empty((ijk,16,192,192,2), dtype='float32')
        selected_patches[:,:,:,:,0] = np.reshape(patches_vol1,(ijk,16,192,192))
        selected_patches[:,:,:,:,1] = np.reshape(patches_vol2,(ijk,16,192,192))
    else:
        selected_patches = np.reshape(patches_vol1,(ijk,16,192,192,1)).astype('float32')

    return selected_patches


def predict(
    model: tf.keras.Model,
    patches: np.ndarray
) -> np.ndarray:
    """Runs model inference on patches and reconstructs the full 3D volume.

    The function slides through the patches, predicts the pseudo-Umap (pmu) content,
    and handles overlapping areas by maintaining a counter to calculate a 
    voxel-wise average, reducing edge artifacts.

    Args:
        model (tf.keras.Model): The pre-trained Keras/TensorFlow model.
        patches (np.ndarray): Normalized patches of shape (N, 16, 192, 192, C).

    Returns:
        np.ndarray: The reconstructed 192x192x192 volume.
    """
    # Settings for patch extraction
    h = 16 # Number of slices 
    sh = 2 # Patch stride

    # Container matrices for data and counter for overlapping patches
    predicted_combined = np.zeros((192,192,192))
    predicted_counter = np.zeros((192,192,192))

    # Process a patch at a time
    for p in range(patches.shape[0]):
        from_h = p*sh # Start slice of patch
        predicted = model.predict(np.reshape(patches[p,:,:,:,:],(1,16,192,192,patches.shape[-1]))) # Predict pmu for patch
        predicted[ predicted == np.nan ] = -1 # Can occur, remove so output does not fail, but set to a value that can be searched for
        predicted_combined[from_h:from_h+h,:,:] += np.reshape(predicted,(16,192,192)) # Insert into container
        predicted_counter[from_h:from_h+h,:,:] += 1 # Update counter in area of patch for later average

    predicted_combined = np.divide(predicted_combined,predicted_counter) # Average over overlapping patches
    predicted_combined[ predicted_combined == np.inf ] = 0 # If divide by zero, remove here (should not occur since counter >> 0).
    
    return predicted_combined

def predict_DeepUTE(
    ute1: np.ndarray,
    ute2: np.ndarray,
    version: str = 'VE11P'
) -> np.ndarray | None:
    """End-to-end pipeline for DeepUTE model inference.

    Handles model selection based on scanner software version, patch extraction 
    with specific UTE normalization, and 3D reconstruction.

    Args:
        ute1 (np.ndarray): Echo 1 UTE sequence.
        ute2 (np.ndarray): Echo 2 UTE sequence.
        version (str): Software version ('VE11P' or 'VB20P'). Defaults to 'VE11P'.

    Returns:
        np.ndarray | None: The predicted pUmap volume x 10000, or None if version is unsupported.
    """
    # Load model
    if version == 'VE11P':
        model_h5 = './models/DeepUTE/DeepUTE_VE11P_model1_TF2.h5' # UPDATE MODEL 01-09-2020]
    elif version == 'VB20P':
        model_h5 = './models/DeepUTE/DeepUTE_VB20P_TF2.h5' # CHANGED TO THIS VERSION 06-03-2019]
    else:
        print('Incorrect software version - no model found')
        return None
    
    model = tf.keras.models.load_model(model_h5,compile=False)
    
    # Load all patches
    patches = get_patches_znorm(ute1,ute2,normalize_both=False)
    
    return predict(model,patches)

def predict_DeepDixon(
    inphase: np.ndarray,
    opposedphase: np.ndarray,
    version: str = 'VE11P'
) -> np.ndarray | None:
    """End-to-end pipeline for DeepDixon model inference.

    Specifically uses independent Z-score normalization (normalize_both=True) 
    for the two Dixon phases before patch extraction.

    Args:
        inphase (np.ndarray): In-phase Dixon volume.
        opposedphase (np.ndarray): Opposed-phase Dixon volume.
        version (str): Software version ('VE11P' or 'VB20P').

    Returns:
        np.ndarray | None: The predicted pUmap volume x 10000.
    """
    # Load model
    if version == 'VE11P':
        model_h5 = './models/DeepDixon/DeepDixon_VE11P_model1_TF2.h5' # UPDATE MODEL 01-09-2020]
    elif version == 'VB20P':
        model_h5 = './models/DeepDixon/DeepDixon_VB20P_TF2.h5' # CHANGED TO THIS VERSION 06-03-2019]
    else:
        print('Incorrect software version - no model found')
        return None
    
    model = tf.keras.models.load_model(model_h5,compile=False)
    
    # Load all patches
    patches = get_patches_znorm(inphase,opposedphase,normalize_both=True)
    
    return predict(model,patches)

def predict_DeepT1(
    t1: np.ndarray,
    version: str = 'VE11P'
) -> np.ndarray | None:
    """End-to-end pipeline for DeepT1 model inference.

    Performs single-channel normalization and patch extraction for T1-weighted 
    MPRAGE sequences.

    Args:
        t1 (np.ndarray): T1-weighted anatomical volume.
        version (str): Software version ('VE11P' or 'VB20P').

    Returns:
        np.ndarray | None: The predicted pUmap volume x 10000.
    """
    # Load model
    if version == 'VE11P':
        model_h5 = './models/DeepT1/DeepT1_VE11P_model1_TF2.h5' # UPDATE MODEL 01-09-2020]
    elif version == 'VB20P':
        model_h5 = './models/DeepT1/DeepT1_VB20P_TF2.h5' # CHANGED TO THIS VERSION 06-03-2019]
    else:
        print('Incorrect software version - no model found')
        return None
    
    model = tf.keras.models.load_model(model_h5,compile=False)
    
    # Load all patches
    patches = get_patches_znorm(t1)
    
    return predict(model,patches)
