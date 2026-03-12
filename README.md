# DeepMRAC - Inference and Evaluation Pipeline

DeepMRAC is a deep learning-based framework for generating pseudo-CT (pCT) attenuation correction maps from various MRI sequences (Dixon, T1w MPRAGE, and UTE).

This implementation utilizes the original models and methodology described in:
Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the training group size. Neuroimage. 2020;222:117221. doi:10.1016/j.neuroimage.2020.117221 .

![Example pseudoCT images](/images/figure2.png)

All versions are implemented for VB20P and VE11P in seperate models.

## Overview 

The pipeline automates the entire workflow from raw DICOM data to clinical-ready attenuation maps:

DICOM Sorting: Organizes raw files by Instance Number.

NIfTI Conversion: Standardizes data using dicom2nifti.

Resampling: Isotropic resampling to a 192x192x192 matrix (1.56mm voxels).

Deep Learning Inference: 3D U-Net prediction using a sliding window (16-slice patches).

DICOM Export: Re-projection to native geometry using the Umap as a header template.

Quality Assessment: Automatic calculation of MAE, PSNR, SSIM, and Dice coefficients.

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

With the `deepmrac_env` environment activated, you can execute the following CLI commands from any directory.Choose the command corresponding to the model you wish to evaluate:

### T1-weighted (MPRAGE) Model

```
process-deep-t1 \
    --t1_path your_t1_folder \
    --umap_path your_umap_folder \
    --output_folder output \
    --save_predictions False \
    --verbose True \
    --version VE11P #could be VB20P
```


### UTE (Ultra-short Echo Time) Model

```
process-deep-ute \
    --ute1_path your_ute1_folder \
    --ute2_path your_ute2_folder \
    --umap_path your_umap_folder \
    --output_folder output \
    --save_predictions False \
    --verbose True \
    --version VE11P #could be VB20P
```

### Dixon (In-phase / Opposed-phase) Model

```
process-deep-dixon \
    --inphase_path your_inphase_folder \
    --opposedphase_path your_opposedphase_folder\
    --umap_path your_umap_folder \
    --output_folder output \
    --save_predictions False \
    --verbose True \
    --version VE11P #could be VB20P
```


## Results and Metrics

Upon completion, the pipeline:

Generates a new DICOM series in the specified output folder.

Prints quality metrics to the console.

Automatically appends results to a summary file using Pandas: output/all_metrics.csv.

