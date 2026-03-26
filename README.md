# DeepMRAC - Inference and Evaluation Pipeline

DeepMRAC is a deep learning-based framework for generating synthetic attenuation correction maps from various MRI sequences (Dixon, T1w MPRAGE, and UTE).

This implementation utilizes the original models and methodology described in:
Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the training group size. Neuroimage. 2020;222:117221. doi:10.1016/j.neuroimage.2020.117221 .

![Example pseudoCT images](/images/figure2.png)

All versions are implemented for VB20P and VE11P in seperate models.

## Overview 

The pipeline provides a fully automated end-to-end workflow, transforming raw anatomical MRI data into clinical-ready PET attenuation maps (sUmaps):

DICOM Orchestration: Automatically identifies and organizes raw RMI and Umap (DICOM or Interfile) datasets by Instance Number to ensure spatial continuity.

Standardized Pre-processing: * NIfTI Conversion: Converts vendor-specific DICOM data into a standardized NIfTI format.

Isotropic Resampling: Resamples data to a unified 192×192×192 matrix (1.56mm isotropic voxels) to match the model's receptive field.

Deep Learning Inference: Executes a 3D U-Net prediction using a sliding window approach (16-slice patches) to generate a synthetic CT (sCT) volume.

Clinical Integration: * Inverse Transformation: Reverts the orientation and resamples the prediction back to the native Umap geometry.

Header Re-projection: Wraps the predicted volume into the original DICOM/Interfile metadata (template) for seamless PACS or workstation integration.

Quality Assurance: If a CT or CTAC is provided, automatically computes and saves global and tissue-specific metrics (MAE, PSNR, SSIM, and Dice) between the synthetic Umap and the CT one.

## Installation

### 1- Clone the repository

```
git clone git@github.com:annegab1804/DeepMRAC.git
```

### 2- Download the models

Download the models from https://drive.google.com/drive/folders/1WJS7n2torSBFBCCnoJhKN9SalYKSUdlM?usp=sharing and put them in the `models/` folder. 
It should look like this: 
```text
DeepMRAC/
└── models/
    ├── DeepDixon/
    │   ├── DeepDixon_VB20P_TF2.h5
    │   └── DeepDixon_VE11P_model1_TF2.h5
    ├── DeepT1/
    │   ├── DeepT1_VB20P_TF2.h5
    │   └── DeepT1_VE11P_model1_TF2.h5
    └── DeepUTE/
        ├── DeepUTE_VB20P_TF2.h5
        └── DeepUTE_VE11P_model1_TF2.h5
```

### 3- Create a Conda environment 

It is highly recommended to use a dedicated environment to manage dependencies with python 3.12.
```
# Create the environment
conda create -n deepmrac_env python=3.12

# Activate the environment
conda activate deepmrac_env
```

### 4- Install the dependencies

Using the provided `setup.py`, you can install the project and all its dependencies in one step. Using the -e flag (editable mode) allows you to modify the code without needing to reinstall.
```
# From the root of the repository
pip install -e .
```

## Running the scripts

With the `deepmrac_env` environment activated, you can execute the following CLI commands from any directory. Choose the command corresponding to the model you wish to evaluate. RMI folder must contrain Dicom files. Umap folder must contain Dicom or Interfile files. Another CT can be provided (as a Dicom folder or directly a NIftI image) if you want to compare the synthetic CT results with it. You must provide the X-ray tube voltages of this CT scanner (kvp). 

### T1-weighted (MPRAGE) Model

```
process-deep-t1 \
    --t1_path your_t1_folder \
    --umap_path your_umap_folder \
    --ct_path your_ct_path \
    --ct_kvp 120 \
    --output_folder output \
    --verbose True \
    --version VE11P #could be VB20P
```


### UTE (Ultra-short Echo Time) Model

```
process-deep-ute \
    --ute1_path your_ute1_folder \
    --ute2_path your_ute2_folder \
    --umap_path your_umap_folder \
    --ct_path your_ct_path \
    --ct_kvp 120 \
    --output_folder output 
    --verbose True \
    --version VE11P #could be VB20P
```

### Dixon (In-phase / Opposed-phase) Model

```
process-deep-dixon \
    --inphase_path your_inphase_folder \
    --opposedphase_path your_opposedphase_folder\
    --umap_path your_umap_folder \
    --ct_path your_ct_path \
    --ct_kvp 120 \
    --output_folder output \
    --verbose True \
    --version VE11P #could be VB20P
```


## Results and Metrics

Upon completion, the pipeline:

Generates a new DICOM series or interfile file in the specified output folder.

Plots axial, coronal and sagittal views of the RMI inputs, the predicted Umap, the CT-Umap used as a template and the reference CT converted into a Umap if provided.

If another CT is provided, prints quality metrics to the console and appends results to a summary file using Pandas: output/all_metrics.csv.

