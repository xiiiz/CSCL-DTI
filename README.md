# CSCL-DTI
##### CSCL-DTI:predicting drug-target interaction through cross-view and self-supervised contrastive learning.
  - [Overview](#overview)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
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

## Description of data files
1. `data/gpcr-train.txt/` is the drug-target interaction training data of the GPCR data set. `data/gpcr-test.txt/` is the drug target interaction test data of the GPCR data set.
    ```
    drug	                                                  target	label	   
    CC(=O)Nc1cc(Cl)cc2cc(C(=O)N3CCN(Cc4ccc(F)cc4)CC3C)oc12	METPNTTEDYDTTTEFDYGDATPCQKVNERAFGAQLLPPLYSLVFVIGLVGNILVVLVLVQYKRLKNMTSIYLLNLAISDLLFLFTLPFWIDYKLKDDWVFGDAMCKILSGFYYTGLYSEIFFIILLTIDRYLAIVHAVFALRARTVTFGVITSIIIWALAILASMPGLYFSKTQWEFTHHTCSLHFPHESLREWKLFQALKLNLFGLVLPLLVMIICYTGIIKILLRRPNEKKSKAVRLIFVIMIIFFLFWTPYNLTILISVFQDFLFTHECEQSRHLDLAVQVTEVIAYTHCCVNPVIYAFVGERFRKYLRQLFHRRVAVHLVKWLPFLSVDRLERVSSTSPSTGEHELSAGF 1
	10000	5
    ...     ...     ...     ...
    ```
    where `drug` is the PubChem Compound ID (CID) of the drug, `protein` is the kinbase name, `Kd` is the binding affinity in nM, and `y` is the log-transformed binding affinity (see paper). The `davis_protein2pdb.yaml` file contains the mapping from a kinase name to its representative PDB structure ID. The `davis_cluster_id50_cluster.tsv` is the [clustering output file](https://github.com/soedinglab/MMseqs2/wiki#cluster-tsv-format) of the MMseqs2 clustering algorithm with a sequence identity cutoff 50% (the first column contains the representative sequences and the second column contains cluster members).
2. `data/KIBA/` is the KIBA dataset of kinase-inhibitor binding affinity. The files in this directory are similar to those in `data/DAVIS/`. In the `kiba_data.tsv` file, the `drug` column contains CHEMBL IDs of the drugs, and the `protein` column contains UniProt IDs of the kinases.
3. `data/structure` contains several structure files. 
    - The `pockets_structure.json` contains the PDB structure data of representative kinase structures. The file is in JSON format where the key is the PDB ID, and the value is the corresponding PDB structure data in a dictionary format, including the following fields: `name` (kinase name), `UniProt_id`, `PDB_id`, `chain` (chain ID in the PDB structure), `seq` (pocket protein sequence), `coords` (coordinates of the N, CA, C, O atoms of residues in the pocket). The `coords` is a dictionary with four fields, and each is a list of xyz coordinates of the N/CA/C/O atom in each residue, i.e., `coords={'N':[[x, y, z], ...]], 'CA': [...], 'C': [...], O: [...]}`
    - The `davis_moil3d_sdf` and `kiba_moil3d_sdf` are diretories that contain the 3D structure (SDF format) of molecules in the Davis and KIBA datasets.
4. `data/esm1b` contains the pre-computed protein sequence embeddings by the ESM-1b model. The embeddings were saved as PyTorch tensors in the `.pt` format. 

## Usage

python main.py

