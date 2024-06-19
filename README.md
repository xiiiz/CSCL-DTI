# CSCL-DTI
##### CSCL-DTI:predicting drug-target interaction through cross-view and self-supervised contrastive learning.
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Quick Example](#quick-example)
  - [Other usages](#other-usages)
  - [Description of data files](#description-of-data-files)
  - [Citation](#citation)
  - [Contact](#contact)

## Overview
CSCL-DTI employs a hybrid contrastive learning architecture to enhance representation learning for predicting DTI.
![HCL-DTI](https://github.com/xiiiz/HCL-DTI/assets/105473770/79776a68-9e1c-4b0c-bbc3-7a8ea358f15c)

## Installation
```bash
git clone https://github.com/xiiiz/CSCL-DTI.git 
cd CSCL-DTI
```
## Dependencies
This package is tested with Python 3.7 and CUDA 11.4 on Ubuntu 18.04.5, with access to an Nvidia A100 GPU (80GB RAM). Run the following to create a conda environment and install the required Python packages. 
```bash
conda create -n CSCL-DTI python=3.7
conda activate CSCL-DTI
```
Running the above lines of `conda install` should be sufficient to install all  CSCL-DTI's required packages (and their dependencies). Specific versions of the packages we tested were listed in `requirements.txt`.

## Requirements

- ##### python == 3.7

- ##### numpy==1.21.6

- ##### pandas==1.1.4

- ##### rdkit==2023.3.2

- ##### scikit_learn==1.0.2

- ##### subword_nmt==0.3.8

- ##### torch==1.13.1

- ##### transformers==3.4.0

## File description

- ##### data_preprocess: Process the data to get the input of the model

- ##### main.py: start file for model training

- ##### model.py: the construction of the neural network

## Usage

python main.py

