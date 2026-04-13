# DeepMRAC - Inference and Evaluation Pipeline

DeepMRAC is a deep learning-based framework for generating synthetic attenuation correction maps from various MRI sequences (Dixon, T1w MPRAGE, and UTE).

This implementation utilizes the original models and methodology described in:
Ladefoged CN, Hansen AE, Henriksen OM, et al. AI-driven attenuation correction for brain PET/MRI: Clinical evaluation of a dementia cohort and importance of the training group size. Neuroimage. 2020;222:117221. doi:10.1016/j.neuroimage.2020.117221 .

![Example pseudoCT images](/images/figure2.png)

All versions are implemented for VB20P and VE11P in separate models.

## Overview 

The pipeline provides a fully automated end-to-end workflow, transforming raw anatomical MRI data into clinical-ready PET attenuation maps (sUmaps):

DICOM Orchestration: Automatically identifies and organizes raw MRI and Umap (DICOM or Interfile) datasets by Instance Number to ensure spatial continuity.

Standardized Pre-processing: * NIfTI Conversion: Converts vendor-specific DICOM data into a standardized NIfTI format.

Isotropic Resampling: Resamples data to a unified 192×192×192 matrix (1.56mm isotropic voxels) to match the model's receptive field.

Deep Learning Inference: Executes a 3D U-Net prediction using a sliding window approach (16-slice patches) to generate a synthetic CT (sCT) volume.

Clinical Integration: * Inverse Transformation: Reverts the orientation and resamples the prediction back to the native Umap geometry.

Header Re-projection: Wraps the predicted volume into the original DICOM/Interfile metadata (template) for seamless PACS or workstation integration.

Quality Assurance: If a CT or CTAC is provided, automatically computes and saves global and tissue-specific metrics (MAE, PSNR, SSIM, and Dice) between the synthetic Umap and the CT one.

To compare the reconstructed PET images using the synthetic umaps to the original PET images, two scripts are also provided. 

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

With the `deepmrac_env` environment activated, you can execute the following CLI commands from any directory. Choose the command corresponding to the model you wish to evaluate. MRI folder must contrain Dicom files. Umap folder must contain Dicom or Interfile files. Another CT can be provided (as a Dicom folder or directly a NIftI image) if you want to compare the synthetic CT results with it. You must provide the X-ray tube voltages of this CT scanner (kvp). 

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

Plots axial, coronal and sagittal views of the MRI inputs, the predicted Umap, the CT-Umap used as a template and the reference CT converted into a Umap if provided.

If another CT is provided, prints quality metrics to the console and appends results to a summary file using Pandas: output/all_metrics.csv.

## Plots to compare SUV 

To evaluate the PET images reconstructed with synthetic Umaps (uhr) against a reference method (hrrt), we provide a dedicated evaluation workflow.

### Prerequisites: FSL Installation
The following steps require FSL (FMRIB Software Library). It must be installed on your system (Linux or macOS) and available in your $PATH. Note: FSL cannot be installed via pip. Please refer to the official FSL installation guide.

### Spatial Pre-processing

Before running the evaluation script, you must align the PET images to the MRI space and generate anatomical masks. Run these commands for each patient:

```
# 1. Brain Extraction (Skull-stripping)
bet T1_weighted.nii.gz brain.nii.gz -f 0.5 -g 0

# 2. Subcortical Segmentation (to create the ROI masks)
run_first_all -i brain.nii.gz -o output_segmentation

# 3. Coregister PET images to MRI space using FLIRT (6 DOF)
# Align synthetic Umap PET (uhr)
flirt -in uhr_suv.nii.gz -ref T1_weighted.nii.gz -out uhr_suv_in_MRI.nii.gz -omat pet2mri.mat -dof 6

# Align reference PET (hrrt)
flirt -in hrrt_suv.nii.gz -ref T1_weighted.nii.gz -out hrrt_suv_in_MRI.nii.gz -applyxfm -init pet2mri.mat -dof 6
```

### Organizing for Batch Analysis

Ensure your data is organized as follows to allow the script to iterate through all subjects:

```
patient_folder/
    ├── patient_01/
    │   ├── uhr_suv_in_MRI.nii.gz   <-- uhr_name
    │   ├── hrrt_suv_in_MRI.nii.gz  <-- hrrt_name
    │   └── output_all_fast_firstseg.nii.gz  <-- seg_name
    ├── patient_02/
    │   ├── uhr_suv_in_MRI.nii.gz
    │   ├── hrrt_suv_in_MRI.nii.gz
    │   └── output_all_fast_firstseg.nii.gz
    └── patient_03/
        ├── uhr_suv_in_MRI.nii.gz
        ├── hrrt_suv_in_MRI.nii.gz
        └── output_all_fast_firstseg.nii.gz
```

### Generate Evaluation Plots

Once the files are aligned, run the following command to generate the statistical analysis (Bland-Altman, Percentage Difference, and Violin plots) for the different Regions of Interest (ROIs).

```
create-suv-plots \
    --input_folder patient_folder \
    --output_folder suv_plots \
    --uhr_name  uhr_suv_in_MRI.nii.gz \
    --hrrt_name hrrt_suv_in_MRI.nii.gz \
    --seg_name output_all_fast_firstseg.nii.gz \
```