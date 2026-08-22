"""
Export MAE pseudo-absences -> hein_merged_patches_mae.csv (fanogan format).

Pipeline per patch (must match how the classifier was trained):
  raw .tif (20 band) -> drop band 20 -> per-band normalize -> MAE reconstruct
  -> residual |input - recon| -> classifier -> P(abnormal)

Then: rank patches by P(abnormal), walk from most anomalous down, collect
BACKGROUND pixels (band_20 == 0) with their RAW 19-band values until N reached.
Stack presence rows. Write CSV with columns band_1..band_20.

Usage:
  python export_pseudoabsence.py \
    --mae   //Users/justin/Desktop/MAE_corn_pseduoabsence/outputs_32x32_normalized/checkpoint-99.pth \
    --clf   /Users/justin/Desktop/MAE_corn_pseduoabsence/output_folder/checkpoint-0.pth \
    --data  /Users/justin/Desktop/MAE_corn_pseduoabsence/data/TestingPatches_3000_32x32 \
    --presence /Users/justin/Desktop/MAE_corn_pseduoabsence/data/presencePoints-corn-kansas.csv \
    --n 8000 \
    --out   /Users/justin/Desktop/MAE_corn_pseduoabsence/hein_merged_patches_mae_PLAD_32x32.csv
"""
import sys
import types
import collections.abc

# Compatibility fix for timm==0.3.2 with modern PyTorch
torch_six = types.ModuleType("torch._six")
torch_six.container_abcs = collections.abc
torch_six.string_classes = (str,)
torch_six.int_classes = (int,)
sys.modules["torch._six"] = torch_six

import timm

import argparse, glob, os
import numpy as np
import pandas as pd
import torch
import tifffile
from tqdm import tqdm

import models_mae
import models_vit

N_BANDS = 19

ap = argparse.ArgumentParser()
ap.add_argument("--mae", required=True)
ap.add_argument("--clf", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--presence", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--mask-ratio", type=float, default=0.75)
ap.add_argument("--num-trials", type=int, default=4)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print("device:", device)


def load_raw(path):
    """(20,H,W) -> (H,W,20) float. Keeps all 20 bands (band 20 = presence)."""
    img = tifffile.imread(path).astype(np.float32)
    if img.shape[0] == 20:                 # (C,H,W)
        img = np.transpose(img, (1, 2, 0))
    return img                              # (H,W,20)


def per_band_norm(x19):
    m = x19.mean(axis=(0, 1), keepdims=True)
    s = x19.std(axis=(0, 1), keepdims=True)
    return (x19 - m) / (s + 1e-8)


def load_mae():
    m = models_mae.mae_vit_base_patch16(img_size=32, patch_size=2, in_chans=N_BANDS)
    ck = torch.load(args.mae, map_location="cpu", weights_only=False)
    print("MAE:", m.load_state_dict(ck["model"], strict=False))
    return m.to(device).eval()


def load_clf():
    # m = models_vit.vit_base_patch2(num_classes=2, drop_path_rate=0.0,
    #                                global_pool=True, img_size=32)
    m = models_vit.vit_base_patch2(num_classes=2, drop_path_rate=0.0,
                                   global_pool=True, img_size=32)
    ck = torch.load(args.clf, map_location="cpu", weights_only=False)
    print("CLF:", m.load_state_dict(ck["classifier"], strict=False)) #model orginally trained with 2 classes, but we only use the abnormal class (1) for scoring
    return m.to(device).eval()


def reconstruct(mae, x19):
    """x19: (H,W,19) normalized -> reconstruction (H,W,19), averaged over trials."""
    x = torch.tensor(x19).unsqueeze(0)
    x = torch.einsum("nhwc->nchw", x).float().to(device)
    acc = None
    for idx in range(args.num_trials):
        _, pred, _ = mae(x, mask_ratio=args.mask_ratio, idx_masking=idx, is_testing=False)
        rec = mae.unpatchify(pred)
        rec = torch.einsum("nchw->nhwc", rec)[0]
        acc = rec if acc is None else acc + rec
    return (acc / args.num_trials).detach().cpu().numpy()


def score_patch(mae, clf, x19_norm):
    """residual -> classifier -> P(abnormal)."""
    rec = reconstruct(mae, x19_norm)
    resid = np.abs(x19_norm - rec)                       # (H,W,19)
    t = torch.tensor(resid).unsqueeze(0)
    t = torch.einsum("nhwc->nchw", t).float().to(device)
    with torch.no_grad():
        out = clf(t)
        p = torch.softmax(out, dim=1)[0, 1].item()     # P(abnormal) torch.softmax(out, dim=1)[0,1].item()
    return p


def main():
    mae, clf = load_mae(), load_clf()
    paths = sorted(glob.glob(os.path.join(args.data, "*.tif")))
    print(f"{len(paths)} patches")

    # score every patch
    scored = []
    for p in tqdm(paths, desc="scoring"):
        raw = load_raw(p)                       # (H,W,20)
        x19 = per_band_norm(raw[:, :, :N_BANDS])
        s = score_patch(mae, clf, x19)
        scored.append((s, p))

    # rank most anomalous first
    scored.sort(key=lambda t: t[0], reverse=True)

    # walk down, collect BACKGROUND pixels' RAW 19-band values
    rows = []
    for s, p in scored:
        raw = load_raw(p)                       # (H,W,20)
        presence = raw[:, :, 19]                # band 20
        bg = np.argwhere(presence == 0)         # background pixel coords
        for (r, c) in bg:
            rows.append(raw[r, c, :N_BANDS])    # raw 19 bands
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break

    print(f"collected {len(rows)} pseudo-absence pixels")
    pa = pd.DataFrame(rows, columns=[f"band_{i}" for i in range(1, 20)])
    pa["band_20"] = 0

    # presence rows
    pres = pd.read_csv(args.presence)
    if "band_20" not in pres.columns:
        pres["band_20"] = 1

    merged = pd.concat([pres, pa], ignore_index=True)
    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(pres)} presence + {len(pa)} pseudo-absence = {len(merged)} rows")


if __name__ == "__main__":
    main()
