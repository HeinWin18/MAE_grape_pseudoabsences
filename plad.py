import torch
import torch.nn as nn
import torch.nn.functional as F

# Your MAE-side ViT lives in models_vit (same repo as extract_reconstructions.py).
# We import it lazily so this file still imports even when models_vit isn't on the
# path (e.g. for quick tests); set USE_VIT=False to fall back to the conv classifier.
try:
    import models_vit
    _HAS_VIT = True
except Exception:
    _HAS_VIT = False


class Perturbator(nn.Module):
    """Per-band learned nudge over a 19-band raster patch.
    Outputs alpha (multiplicative) and beta (additive), one scalar per band,
    broadcast over H,W."""
    def __init__(self, in_chans=19, device=None):
        super().__init__()
        self.in_chans = in_chans
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 100, 3, padding=1), nn.Tanh(),
            nn.Conv2d(100, 100, 3, padding=1), nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(100, 2 * in_chans),
        )
        if device is not None:
            self.to(device)

    def forward(self, x):
        out = self.net(x)
        alpha = out[:, :self.in_chans].view(-1, self.in_chans, 1, 1)
        beta = out[:, self.in_chans:].view(-1, self.in_chans, 1, 1)
        return alpha, beta


class ViTClassifier(nn.Module):
    """Wraps your vit_base_patch1 to output P(abnormal) in [0,1].
    num_classes=1 + sigmoid so it plugs into the paper's BCE loss unchanged."""
    def __init__(self, img_size=32, in_chans=19, device=None):
        super().__init__()
        self.vit = models_vit.vit_base_patch1(
            img_size=img_size, num_classes=1, global_pool=True,
        )
        if device is not None:
            self.to(device)

    def forward(self, r):
        logits = self.vit(r)              # (B,1)
        return torch.sigmoid(logits)


class ConvClassifier(nn.Module):
    """Fallback classifier if models_vit isn't importable (for local tests)."""
    def __init__(self, in_chans=19, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        if device is not None:
            self.to(device)

    def forward(self, r):
        return self.net(r)


class PLAD(nn.Module):
    """Joint perturbator + classifier over MAE residuals.

    Given presence patch x and frozen-MAE reconstruction x_hat:
      normal residual   r  = |x - x_hat|
      perturbed recon   x' = alpha * x_hat + beta      (the abnormal patch)
      abnormal residual r' = |x - x'|
    """
    def __init__(self, img_size=32, in_chans=19, device=None, use_vit=True):
        super().__init__()
        self.in_chans = in_chans
        self.perturbator = Perturbator(in_chans, device)
        if use_vit and _HAS_VIT:
            self.classifier = ViTClassifier(img_size, in_chans, device)
        else:
            self.classifier = ConvClassifier(in_chans, device)

    def forward(self, x, x_hat):
        r = (x - x_hat).abs()
        alpha, beta = self.perturbator(x_hat)
        x_pert = alpha * x_hat + beta            # abnormal patch (viewable)
        r_pert = (x - x_pert).abs()
        return self.classifier(r), self.classifier(r_pert), alpha, beta, x_pert