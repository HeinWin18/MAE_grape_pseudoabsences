#!/usr/bin/env python3
"""
AnoVAEGAN inference + pseudo-absence extraction.

Input : folder of 32x32x20 .tif testing patches
        bands 1-19 = environmental variables, band 20 = presence (0/1)
Model : trained AnoVAEGAN checkpoint {"encoder":..., "decoder":...}
Score : mean pixelwise L1 reconstruction residual over the 19 bands
Output: top-K highest-scoring valid pixels as MaxEnt pseudo-absences (presence=0)
"""

import argparse, csv, glob, os
import numpy as np
import tifffile
import torch
import torch.nn as nn

C = 19
P = 32
Z = 128
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, c=C, z=Z):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
        )
        self.mu = nn.Conv2d(128, z, 3, 1, 1)
        self.logvar = nn.Conv2d(128, z, 3, 1, 1)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, c=C, z=Z):
        super().__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(z, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, c, 3, 1, 1),
        )

    def forward(self, z):
        return self.body(z)


def reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def load_model(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location=DEVICE)
    enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
    enc.load_state_dict(ck["encoder"])
    dec.load_state_dict(ck["decoder"])
    enc.eval(); dec.eval()
    return enc, dec


def normalize_patch(image):
    """Per-patch, per-band z-score, matching AnoVAEGAN training."""
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    out = (image - mean) / (std + 1e-8)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def score_patch(enc, dec, img_norm):
    """Return (32,32) map of mean-over-bands L1 residual."""
    x = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    x_hat = dec(reparameterize(*enc(x)))
    return (x - x_hat).abs().mean(dim=1)[0].cpu().numpy()


def run(testpath, checkpoint_path, top_k, output_csv, exclude_presence):
    print(f"device={DEVICE}  testpath={testpath}  top_k={top_k}")
    enc, dec = load_model(checkpoint_path)

    files = sorted(glob.glob(os.path.join(testpath, "*.tif")))
    if not files:
        raise RuntimeError(f"No .tif files in {testpath}")
    print(f"found {len(files)} patches")

    scores_all, bio_all = [], []
    for k, fp in enumerate(files, 1):
        try:
            arr = tifffile.imread(fp).astype(np.float32)
            if arr.ndim != 3:
                continue
            if arr.shape == (C + 1, P, P):        # bands-first -> H,W,C
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape != (P, P, C + 1):
                print(f"[skip] {os.path.basename(fp)}: shape {arr.shape}")
                continue

            img_raw = arr[:, :, :C]
            presence = arr[:, :, -1]

            # valid = finite environmental values; optionally exclude presence cells
            valid = np.all(np.isfinite(img_raw), axis=-1)
            if exclude_presence:
                valid &= (presence == 0)
            if not np.any(valid):
                continue

            smap = score_patch(enc, dec, normalize_patch(img_raw))
            r, c = np.where(valid)
            scores_all.append(smap[r, c])
            bio_all.append(img_raw[r, c, :])
        except Exception as e:
            print(f"[err] {os.path.basename(fp)}: {e}")
            continue
        if k == 1 or k % 50 == 0 or k == len(files):
            print(f"[{k}/{len(files)}] processed")

    scores = np.concatenate(scores_all)
    bios = np.concatenate(bio_all)
    print(f"valid pixel candidates: {len(scores):,}")
    if len(scores) < top_k:
        raise RuntimeError(f"only {len(scores):,} valid pixels, need {top_k:,}")

    order = np.argsort(-scores)[:top_k]
    sel_bio = bios[order]
    sel_scores = scores[order]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"bio{i}" for i in range(1, C + 1)] + ["presence"])
        for row in sel_bio:
            w.writerow(row.astype(float).tolist() + [0])

    print(f"selected {len(order):,} pseudo-absences")
    print(f"score range: {sel_scores.min():.6f} .. {sel_scores.max():.6f} (mean {sel_scores.mean():.6f})")
    print(f"saved -> {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract pseudo-absences using trained AnoVAEGAN.")
    ap.add_argument("--testpath", required=True, help="folder of 32x32x20 .tif patches")
    ap.add_argument("--checkpoint", required=True, help="path to anovaegan.pth")
    ap.add_argument("--top_k", type=int, default=8000)
    ap.add_argument("--output", default="pseudo_absences_anovaegan.csv")
    ap.add_argument("--exclude-presence", action="store_true",
                    help="exclude cells where band 20 == 1 (only sample from background)")
    args = ap.parse_args()
    run(args.testpath, args.checkpoint, args.top_k, args.output, args.exclude_presence)