"""
anovaegan_torch.py

PyTorch reimplementation of AnoVAEGAN (Baur et al.) as a pseudo-absence baseline,
using the SAME .tif dataloader and preprocessing as the MAE training pipeline.

  - Input = original .tif patches (e.g. TrainingPatches_300_32x32).
  - Dataloader is CropPatchDataset, copied from the MAE pipeline (tifffile,
    19-band slice, per-band z-score, NaN guard, permute to C,H,W).
  - Trained on presence patches only. NO validation split (train-only, like the MAE).

Faithful to AnoVAEGAN:
  - variational encoder (mu, logvar) + reparameterization
  - decoder / generator
  - IMAGE discriminator (real x vs reconstructed x_hat), WGAN-GP
  - three alternating steps per batch: VAE -> generator -> critic xN
  - anomaly score at inference = L1 reconstruction residual

Usage:
    python3 anovaegan_torch.py --data /path/TrainingPatches_300_32x32 --out anovaegan_ckpt
"""

import argparse, os
import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

C = 19
P = 32
Z = 128
BATCH = 64
EPOCHS = 1600
N_CRITIC = 5
GP_LAMBDA = 10.0
KL_WEIGHT = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---- dataloader copied from the MAE pipeline (CropPatchDataset) ----
class CropPatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".tif")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = tifffile.imread(self.files[idx]).astype(np.float32)
        image = image[:, :, :19]
        mean = image.mean(axis=(0, 1), keepdims=True)
        std = image.std(axis=(0, 1), keepdims=True)
        image = (image - mean) / (std + 1e-8)

        image = torch.from_numpy(image)

        if torch.isnan(image).any():
            print("NaN found in:", self.files[idx])
            raise ValueError("NaN in input patch")

        image = image.permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)

        return image, 0


# ---- networks ----
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


class Discriminator(nn.Module):
    def __init__(self, c=C):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.body(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def gradient_penalty(D, real, fake):
    a = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    inter = (a * real + (1 - a) * fake).requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(d_inter, inter,
                                grad_outputs=torch.ones_like(d_inter),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train(data_folder, out_dir=None):
    print(f"training on {DEVICE}")
    loader = DataLoader(CropPatchDataset(data_folder), batch_size=BATCH,
                        shuffle=True, drop_last=True)

    enc, dec, D = Encoder().to(DEVICE), Decoder().to(DEVICE), Discriminator().to(DEVICE)
    opt_vae = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                               lr=1e-3, betas=(0.5, 0.9))
    opt_g = torch.optim.Adam(dec.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(EPOCHS):
        ep_vae = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)

            mu, logvar = enc(x)
            x_hat = dec(reparameterize(mu, logvar))
            rec = (x - x_hat).abs().mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss_vae = rec + KL_WEIGHT * kl
            opt_vae.zero_grad(); loss_vae.backward(); opt_vae.step()

            mu, logvar = enc(x)
            x_hat = dec(reparameterize(mu, logvar))
            loss_g = -D(x_hat).mean()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            for _ in range(N_CRITIC):
                with torch.no_grad():
                    mu, logvar = enc(x)
                    x_hat = dec(reparameterize(mu, logvar))
                loss_d = D(x_hat).mean() - D(x).mean() + GP_LAMBDA * gradient_penalty(D, x, x_hat)
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            ep_vae += loss_vae.item()

        print(f"EPOCH {epoch+1}: vae_loss={ep_vae/len(loader):.4f}")

        # Save checkpoint every 100 epochs
        if out_dir is not None and (epoch + 1) % 100 == 0:
            os.makedirs(out_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                out_dir,
                f"anovaegan_epoch_{epoch+1}.pth"
            )

            torch.save({
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "discriminator": D.state_dict(),
                "opt_vae": opt_vae.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch + 1,
            }, checkpoint_path)

            print(f"checkpoint saved -> {checkpoint_path}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "anovaegan.pth")
        torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict()}, path)
        print(f"saved -> {path}")

    return enc, dec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of .tif presence patches")
    ap.add_argument("--out", default=None, help="dir to save checkpoint")
    args = ap.parse_args()
    train(args.data, args.out)
