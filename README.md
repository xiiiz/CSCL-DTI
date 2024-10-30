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

If you use the code of DeepGS, please cite the paperbelow:

> @inproceedings{lin2024cscl-dti,
  title={CSCL-DTI: predicting drug-target interaction through cross-view and self-supervised contrastive learning},
  author={Lin, Xuan and Zhang, Xi and Yu, Zu-Guo and Long, Yahui and Zeng, Xiangxiang and Yu, Philip S},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2024},
  organization={IEEE}
}

<h1>Citation</h1>

If you use the code of DeepGS, please cite the [paper](http://ecai2020.eu/papers/34_paper.pdf) below:

> @inproceedings{lin2020deepgs,  
title ={DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction},  
author ={Lin, Xuan and Zhao, Kaiqi and Xiao, Tong and Quan, Zhe and Wang, Zhi-Jie and Yu, Philip S},  
booktitle ={24th European Conference on Artificial Intelligence (ECAI)},  
pages ={1--8},  
year ={2020}  
}
