# LGFVM-UNet
This is the official codes for the work "LGFVM-UNet: A Local-Global Fusion Vision Mamba UNet Framework for Medical Image Segmentation".

## Installation

You need to install the necessary packages using the following instructions:

```bash
pip install -r requirements.txt
```

In our experiments, we used Python version 3.10 and CUDA version 11.8.0. We recommend that you also use the same versions.

## Dataset

You can download all the datasets used in our experiments through the following links:
- **Synapse Multi-organ Segmentation Dataset**: https://drive.google.com/file/d/1m9ihuBdgxDp0hlJyIlh94tJqzllo0klt/view?usp=sharing
- **ACDC Dataset**: https://drive.google.com/file/d/1tKuqc-w7ZC54gvlZlkstB0rmfSjiJXVL/view?usp=drive_link
- **ISIC2017 Dataset**: https://drive.google.com/file/d/1jIYYMFItuIqRY8Zfb6rAcdkHc9myAK00/view?usp=sharing
- **ISIC2018 Dataset**: https://drive.google.com/file/d/1JdHvbVq6jfLApgs7l8chro4wHKcw31UL/view?usp=sharing
- **CVC-ClinicDB Colonoscopy Dataset**: https://drive.google.com/file/d/1FqKDdEPxEhxE7EFPNPPaD1PRM3GAWCNA/view?usp=sharing

For the downloaded datasets, simply extract them to the data folder. To select a specific dataset, you need to set the parameter datasets_name in the config_setting_synapse.py file, and fill in the dataset address according to the configuration parameters like datasets_name = 'synapse'.

## Training

Use the following command to train LGFVM-UNet:

```python
python train_synapse.py
```

## Acknowledgements

We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba), [VM-UNet](https://github.com/JCruan519/VM-UNet), and [MSVM-UNet](https://github.com/gndlwch2w/msvm-unet) for their valuable and open-source code.

