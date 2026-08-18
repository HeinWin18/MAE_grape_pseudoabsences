import argparse
import glob
import os.path
import pickle
import rasterio
import hashlib

# Compatibility fix for timm==0.3.2 with modern PyTorch
import sys
import types

torch_six = types.ModuleType("torch._six")
torch_six.container_abcs = collections.abc
torch_six.string_classes = (str,)
torch_six.int_classes = (int,)
sys.modules["torch._six"] = torch_six

import timm

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import models_mae

img_size_choice = 32
patch_size_choice = 2

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'): #mae_vit_large_patch16
    # build model
    img_size = img_size_choice
    patch_size = patch_size_choice

    model = getattr(models_mae, arch)(img_size=img_size, patch_size=patch_size, in_chans=19)
    checkpoint = torch.load(chkpt_dir, weights_only=False, map_location="cpu")
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = model.to(device)
    return model


def get_reconstructions(model_, imgs_, idx):

    x = torch.tensor(imgs_)

    x = torch.einsum('nhwc->nchw', x)
    
    if torch.cuda.is_available():
        x = x.to('cuda')
    elif torch.backends.mps.is_available():
        x = x.to('mps')
    else:
        x = x.to('cpu')

    loss, result, mask = model_(x.float(), mask_ratio=args.mask_ratio, idx_masking=idx, is_testing=False)
    result = model_.unpatchify(result)
    result = torch.einsum('nchw->nhwc', result).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model_.patch_embed.patch_size[0]**2 * model_.in_chans)  # (N, H*W, p*p*3)
    mask = model_.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # MAE reconstruction pasted with visible patches
    im_paste = torch.einsum('nchw->nhwc', x).detach().cpu() * (1 - mask) + result * mask

    return im_paste.numpy()


def get_reconstructions_multi(model_, imgs_):
    num_fwd = args.num_trials
    results = None
    for idx in range(num_fwd):
        result = get_reconstructions(model_, imgs_, idx)
        if results is None:
            results = result
        else:
            results += result

    results = results / num_fwd

    return results



def load_image(img_path_):
    with rasterio.open(img_path_) as src:
        image_np = src.read()

    # rasterio returns:
    # (channels, height, width)

    image_np = np.transpose(
        image_np,
        (1, 2, 0)
    )

    image_np = image_np[:, :, :19]   # drop presence band (band 20)

    image_np = np.float32(image_np)

    return image_np

def get_patch_paths(dataset_path):
    paths = glob.glob(
        os.path.join(dataset_path, "*.tif")
    )

    paths.sort()

    print(f"Found {len(paths)} patches")

    return paths


def visualize(imgs_, reconstructions_, old_reconstructions_, paths_):
    num_imgs = 5
    for (img_, recon_, old_recon_, path_) in zip(imgs_, reconstructions_, old_reconstructions_, paths_):

        plt.subplot(1, num_imgs, 1)
        plt.imshow(img_, cmap='gray')
        plt.subplot(1, num_imgs, 2)
        plt.imshow(recon_, cmap='gray')
        plt.subplot(1, num_imgs, 3)
        plt.imshow(old_recon_, cmap='gray')
        plt.subplot(1, num_imgs, 4)
        plt.imshow(np.abs(img_ - recon_), cmap='gray')
        plt.subplot(1, num_imgs, 5)
        plt.imshow(np.abs(old_recon_ - recon_), cmap='gray')
        plt.show()


# def save(imgs, reconstructions, used_paths, is_abnormal, iter_):
#     base_dir = args.output_folder
#     if is_abnormal:
#         base_dir = os.path.join(base_dir, 'abnormal')
#     else:
#         base_dir = os.path.join(base_dir, 'normal')

#     for (img_, recon_, path_) in zip(imgs, reconstructions, used_paths):
#         if is_abnormal and img_.sum() == 0:
#             continue

#         info_ = {'img': img_, 'recon': recon_}
#         short_filename = os.path.split(path_)[-1][:-4] + f'_{iter_}.pkl'
#         with open(os.path.join(base_dir, short_filename), 'wb') as handle:
#             pickle.dump(info_, handle)

def save(imgs, reconstructions, used_paths, is_abnormal, iter_):

    class_name = 'abnormal' if is_abnormal else 'normal'

    for img_, recon_, path_ in zip(imgs, reconstructions, used_paths):
        if is_abnormal and img_.sum() == 0:
            continue

        filename = os.path.basename(path_)
        h = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        split = 'val' if h % 5 == 0 else 'train'

        base_dir = os.path.join(args.output_folder, split, class_name)

        info_ = {'img': img_, 'recon': recon_}
        short_filename = os.path.splitext(filename)[0] + f'_{iter_}.pkl'

        with open(os.path.join(base_dir, short_filename), 'wb') as handle:
            pickle.dump(info_, handle)


#old normalization
# def process_image(img_):
#     img_ = img_.astype(np.float32)
#     img_ = (img_ - img_.mean()) / (img_.std() + 1e-8)
#     return img_

#new normalization over each patch
def process_image(img_):
    img_ = img_.astype(np.float32)
    # per-band z-score: stats over H,W for each channel independently
    mean = img_.mean(axis=(0, 1), keepdims=True)
    std  = img_.std(axis=(0, 1), keepdims=True)
    img_ = (img_ - mean) / (std + 1e-8)
    return img_


def write_reconstructions(model_mae, paths, is_abnormal: bool = False, iter_: int = 0):

    for start_index in tqdm(range(0, len(paths), args.batch_size)):
        imgs = []
        used_paths = []
        for idx_path in range(start_index, start_index + args.batch_size):
            if idx_path < len(paths):
                path_ = paths[idx_path]
                img_ = load_image(path_)
                img_ = process_image(img_)

                imgs.append(img_)
                used_paths.append(path_)

        imgs = np.array(imgs, np.float32)

        reconstructions = get_reconstructions_multi(model_mae, imgs)

        # visualize(imgs, reconstructions, reconstructions, used_paths)

        save(imgs, reconstructions, used_paths, is_abnormal, iter_=iter_)


def load_model(model_path):
    model_mae = prepare_model(model_path, 'mae_vit_base_patch16')
    return model_mae


parser = argparse.ArgumentParser(description='PyTorch Medical Images')
parser.add_argument('--model-path', type=str)
parser.add_argument('--mask-ratio', type=float)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--output-folder', type=str, required=True)
parser.add_argument('--num-trials', type=int, default=1)
parser.add_argument('--use_val', action='store_true',
                    help='Test on val data.')

parser.add_argument('--test', action='store_true')

parser.set_defaults(use_val=False)

args = parser.parse_args()

assert os.path.exists(args.dataset), f"Dataset path does not exist: {args.dataset}"
"""  
python3 extract_reconstructions.py --dataset=brats --mask-ratio=0.85  \
--model-path=mae_brats_mask_ratio_0.85/checkpoint-1599.pth --batch-size=64 --num-trials=4 \
--output-folder=/media/lili/SSD2/datasets/brats/BraTS2020_training_data/reconstructions/mae_mask_ratio_0.85/val --use_val --test

 
python3 extract_reconstructions.py --dataset=luna16_unnorm --mask-ratio=0.75  \
--model-path=models/mae_luna16_patch_16_mask_ratio_0.75_unnorm/checkpoint-1599.pth --batch-size=64 --num-trials=4 \
--output-folder=/media/lili/SSD2/datasets/luna16/reconstructions/mae_luna16_patch_16_mask_ratio_0.75_unnorm_3_6/train


 
"""
if __name__ == '__main__':
    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.output_folder, split, "normal"), exist_ok=True)
    model_path = args.model_path
    model_mae_ = load_model(model_path)

    # Data
    normal_paths = get_patch_paths(args.dataset)

    write_reconstructions(
        model_mae_,
        paths=normal_paths,
        is_abnormal=False
    )
