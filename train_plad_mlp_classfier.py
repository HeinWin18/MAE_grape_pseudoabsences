import torch
import torch.nn as nn
import torch.nn.functional as F

# Your MAE-side ViT lives in models_vit (same repo as extract_reconstructions.py).
# We import it lazily so this file still imports even when models_vit isn't on the
# path.
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
    """ViT classifier used by the full PLAD model."""
    def __init__(self, img_size=32, in_chans=19, device=None):
        super().__init__()
        self.vit = models_vit.vit_base_patch1(
            img_size=img_size,
            num_classes=1,
            global_pool=True,
        )
        if device is not None:
            self.to(device)

    def forward(self, r):
        logits = self.vit(r)
        return torch.sigmoid(logits)


class MLPClassifier(nn.Module):
    """Simpler MLP classifier for the classifier-complexity ablation.

    The input residual is globally pooled over H,W so that the MLP operates
    directly on the 19 environmental bands rather than using convolutional
    or attention-based spatial processing.
    """
    def __init__(self, in_chans=19, device=None):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.net = nn.Sequential(
            nn.Linear(in_chans, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        if device is not None:
            self.to(device)

    def forward(self, r):
        x = self.pool(r)
        x = torch.flatten(x, 1)
        return self.net(x)


class PLAD(nn.Module):
    """Joint perturbator + classifier over MAE residuals.

    Given presence patch x and frozen-MAE reconstruction x_hat:
      normal residual   r  = |x - x_hat|
      perturbed recon   x' = alpha * x_hat + beta
      abnormal residual r' = |x - x'|
    """
    def __init__(
        self,
        img_size=32,
        in_chans=19,
        device=None,
        use_vit=True
    ):
        super().__init__()

        self.in_chans = in_chans

        self.perturbator = Perturbator(
            in_chans,
            device
        )

        if use_vit and _HAS_VIT:
            self.classifier = ViTClassifier(
                img_size,
                in_chans,
                device
            )
        else:
            self.classifier = MLPClassifier(
                in_chans,
                device
            )

    def forward(self, x, x_hat):
        r = (x - x_hat).abs()

        alpha, beta = self.perturbator(x_hat)

        x_pert = alpha * x_hat + beta

        r_pert = (x - x_pert).abs()

        return (
            self.classifier(r),
            self.classifier(r_pert),
            alpha,
            beta,
            x_pert
        )
