from setuptools import setup, find_packages

setup(
    name="deepmrac",
    version="1.0.0",
    author="Anne-Gabrielle Gibeili",
    description="Deep learning based pseudo-CT generation from MRI (T1, UTE, Dixon) Inference and Evaluation Pipeline",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "tensorflow>=2.16.1",
        "pydicom>=2.4.0",
        "nibabel>=5.2.0",
        "dicom2nifti>=2.4.0",
        "nilearn>=0.10.2",
        "antspyx>=0.4.2",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.22.0",
        "SimpleITK>=2.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "wget"
    ],
    python_requires=">=3.12",
    entry_points={
        'console_scripts': [
            'process-deep-t1=scripts.process_DeepT1_dicom:main',
            'process-deep-ute=scripts.process_DeepUTE_dicom:main',
            'process-deep-dixon=scripts.process_DeepDixon_dicom:main',
            'create-suv-plots=scripts.create_suv_plots:main',
        ],
    },
)