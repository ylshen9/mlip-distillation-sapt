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
- `He-multi/ft/fig5/plot.py`: reproduce the Fig. 5-style transfer comparison.

## Installation

Install the dependencies in your existing Python environment:

```bash
pip install -r requirements.txt

Some workflows require additional packages or separate environments:

- Orb teacher data generation requires `orb-models`.
- MatterSim teacher data generation requires `mattersim`.
- DeepMD baseline requires `deepmd-kit` and the `dp` command-line tool.

## Reproducibility notes

The `Please read.txt` files in each workflow folder describe the intended use of each script and dataset.

## Citation

Citation information will be updated after publication.
