# MAE Crop Training

This repository contains the code required to train the Masked Autoencoder 
(MAE) used for crop environmental patch experiments.

## Model

- Input: 32 × 32 environmental patches
- Environmental bands: 19
- Patch size: 2 × 2
- MAE architecture: ViT-Base
- Masking ratio: 0.85
- Training target: reconstruction of masked environmental patches

## Repository Structure

```text
main_pretrain.py
engine_pretrain.py
models_mae.py
util/
    misc.py
    lr_sched.py
    pos_embed.py
    datasets.py
requirements.txt

Installation
pip install -r requirements.txt
Dataset

The training dataset consists of 32 × 32 TIFF patches.

Each TIFF contains 20 bands:

Bands 1–19: environmental variables
Band 20: presence/absence information

The MAE uses the first 19 environmental bands.

Training

Example:

python main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --input_size 32 \
    --patch_size 2 \
    --mask_ratio 0.85 \
    --epochs 1600 \
    --data_path 
/content/MAE_grape_pseudoabsences/data/TrainingPatches_3000_32x32 \
    --output_dir /content/MAE_grape_pseudoabsences/outputs_32x32

The training command should be adjusted to the dataset location in Google 
Colab.

Purpose

The trained MAE reconstruction is subsequently used in the MAE + PLAD 
pseudo-absence generation pipeline.
