# CSCL-DTI
##### CSCL-DTI:predicting drug-target interaction through cross-view and self-supervised contrastive learning.
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Description of data files](#description-of-data-files)
  - [Usage](#Usage)

## Overview
CSCL-DTI employs a hybrid contrastive learning architecture to enhance representation learning for predicting DTI.
![HCL-DTI](https://github.com/xiiiz/HCL-DTI/assets/105473770/79776a68-9e1c-4b0c-bbc3-7a8ea358f15c)

## Installation
We recommend to create a new environment.

```bash
conda create -n CSCL-DTI python=3.7
conda activate CSCL-DTI
```
This package is tested with Python 3.7 and CUDA 11.4 on Ubuntu 18.04.5. 
```bash
git clone https://github.com/xiiiz/CSCL-DTI.git 
cd CSCL-DTI
pip install -r requirements.txt
```
Running the above lines of `conda install` should be sufficient to install all  CSCL-DTI's required packages (and their dependencies). Specific versions of the packages we tested were listed in `requirements.txt`.

## Description of data files
1. `data/gpcr-train.txt` is the the GPCR training dataset of drug-target interaction. `data/gpcr-test.txt` is the GPCR testing dataset of drug-target interaction.
    ```
    drug	                                                  target	label	   
    CC(=O)Nc1cc(Cl)cc2cc(C(=O)N3CCN(Cc4ccc(F)cc4)CC3C)oc12	METPNTTEDYDTTTEFDYGDATPCQKVNERAFGAQLLPPLYSLVFVIGLVGNILVVLVLVQYKRLKNMTSIYLLNLAISDLLFLFTLPFWIDYKLKDDWVFGDAMCKILSGFYYTGLYSEIFFIILLTIDRYLAIVHAVFALRARTVTFGVITSIIIWALAILASMPGLYFSKTQWEFTHHTCSLHFPHESLREWKLFQALKLNLFGLVLPLLVMIICYTGIIKILLRRPNEKKSKAVRLIFVIMIIFFLFWTPYNLTILISVFQDFLFTHECEQSRHLDLAVQVTEVIAYTHCCVNPVIYAFVGERFRKYLRQLFHRRVAVHLVKWLPFLSVDRLERVSSTSPSTGEHELSAGF 1
    ...     ...     ...   
    ```
    where `drug` is the Simplified Molecular Input Line-Entry System (SMILES) sequence of the drug, `target` is the protein amino acids sequence, `label` is the drug target interaction.
2. `data/human.txt` is the Human dataset of drug-target interaction. The contents in this file is similar to the contents in `data/gpcr-train.txt/`.
3. `data/drugbank-train.csv/` is the the DrugBank training dataset of drug-target interaction. `data/drugbank-test.txt/` is the DrugBank testing dataset of drug-target interaction. The contents in this file is similar to the contents in `data/gpcr-train.txt/`, except that the target is in the first column and the drug is in the second column.

## Usage

```bash
cd dataset
python main.py
```
dataset specifically refers to gpcr, human and drugbank


## Citation

If you use the code of CSCL-DTI, please cite the below:

> @inproceedings{lin2024cscl-dti,  
title ={CSCL-DTI: predicting drug-target interaction through cross-view and self-supervised contrastive learning},  
author ={Lin, Xuan and Zhang, Xi and Yu, Zu-Guo and Long, Yahui and Zeng, Xiangxiang and Yu, Philip S},  
booktitle ={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},  
year ={2024}  
}
