# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pickle
import matplotlib.pyplot as plt
import numpy as np

def is_valid_file_fn(path_):
    return 'pkl' in path_


# def load_fn(path_):
#     with open(path_, 'rb') as handle:
#         dict_ = pickle.load(handle)

#     img_ = dict_["img"]
#     recon_ = dict_["recon"]
#     diff = np.abs((img_ - recon_))[:, :, 0]

#     return PIL.Image.fromarray(diff)

def load_fn(path_):
    with open(path_, 'rb') as handle:
        dict_ = pickle.load(handle)

    img_ = dict_["img"]
    recon_ = dict_["recon"]
    diff = np.abs(img_ - recon_)
    
    diff = torch.from_numpy(diff).permute(2,0,1).float()
    
    return diff

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform, loader=load_fn, is_valid_file=is_valid_file_fn)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    return None