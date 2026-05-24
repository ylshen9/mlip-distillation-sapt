# MLIP Distillation and Fine-Tuning for He--Benzene and PAH Weak Interactions

This repository contains code and supporting data for training and evaluating machine-learning interatomic potentials for weak He--benzene and He--polycyclic aromatic hydrocarbon interactions.

## Repository structure

```text
He-BZ/       He--benzene workflows for distillation, fine-tuning, direct training, SR/LR, and SAPT-adaptive models.
He-multi/    He--multi-ring PAH transfer workflows for coronene, C-coronene, CC-coronene, and CCC-coronene.
```

## Main workflows

### He--benzene

- `He-BZ/generate_data_dist.py`: generate Orb-based distillation data.
- `He-BZ/generate_data_ft.py`: generate fine-tuning datasets from `HeBz_sector_2475.npy`.
- `He-BZ/SAPT/train.py`: train the SAPT-adaptive SR/LR model.
- `He-BZ/SAPT/direct/train.py`: train the direct model without distillation.
- `He-BZ/SAPT/dft/`: DFT-distillation workflow.
- `He-BZ/MLP/`: DeepMD MLP baseline workflow.
- `He-BZ/sr+lr/`: fixed SR/LR model workflow.

### He--multi-ring PAHs

- `He-multi/generate_data.py`: generate transfer datasets using Orb or MatterSim teachers.
- `He-multi/train.py`: train the combined He--PAH model.
- `He-multi/ft/fig5/all.py`: reproduce the Fig. 5-style transfer comparison.

## Installation

Create a Python environment and install the core dependencies:

```bash
conda create -n mlip-distill python=3.10
conda activate mlip-distill
pip install -r requirements.txt
```

Some workflows require additional packages or separate environments:

- Orb teacher data generation requires `orb-models`.
- MatterSim teacher data generation requires `mattersim`.
- DeepMD baseline requires `deepmd-kit` and the `dp` command-line tool.

## Data and model files

Small processed datasets and final model files may be included in this repository. Large training sweeps, repeated checkpoints, and raw simulation outputs should be stored separately, for example in Zenodo, Figshare, or a GitHub Release.

Recommended files to keep outside normal Git history:

- repeated checkpoint folders such as `He-BZ/SAPT/fig4/*/check*/`
- full raw simulation outputs
- very large `.npy`, `.npz`, `.pt`, `.pb`, or `.ckpt` files unless tracked by Git LFS

## Reproducibility notes

The `Please read.txt` files in each workflow folder describe the intended use of each script and dataset. In particular, for He--benzene fine-tuning, `dataset1`--`dataset5` correspond to increasing fine-tuning fractions.

## Citation

Citation information will be updated after publication.
