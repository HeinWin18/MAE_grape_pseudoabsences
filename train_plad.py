import argparse
import glob
import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from plad import PLAD


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ReconDataset(Dataset):
    """Loads NORMAL reconstructions saved by extract_reconstructions.py.
    Each .pkl = {'img': x (H,W,C), 'recon': x_hat (H,W,C)}."""
    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        assert self.paths, f"no .pkl files in {folder}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        x = torch.tensor(d["img"], dtype=torch.float32).permute(2, 0, 1)      # C,H,W
        x_hat = torch.tensor(d["recon"], dtype=torch.float32).permute(2, 0, 1)
        return x, x_hat


def L(pred_normal, pred_pert, alpha, beta) -> Tensor:
    """PLAD loss, exactly as in the paper (Eq. 2), with the 1/n mean on every term:

        L = (1/n) BCE(pred_normal, 0)
          + (1/n) BCE(pred_pert,   1)
          + (lambda/n) ( ||alpha - 1||^2 + ||beta||^2 )
    """
    global lambd
    loss_fn = nn.BCELoss()
    n = pred_pert.shape[0]

    penalty = (torch.norm(alpha - 1) ** 2 + torch.norm(beta) ** 2) / n

    l = loss_fn(pred_pert, torch.ones_like(pred_pert)) + lambd * penalty
    if not use_pretrained_classifier:
        l += loss_fn(pred_normal, torch.zeros_like(pred_normal))
    return l


use_pretrained_classifier = False
BATCH_SIZE = 64
EPOCHS = 1
lambd = 0.01


@torch.no_grad()
def evaluate(plad, loader, device):
    """Held-out AUC: score normal residuals vs. perturbed (abnormal) residuals.
    label 0 = normal, 1 = abnormal, exactly as the classifier is trained.
    Mirrors the roc_auc_score evaluation from the old engine_finetune.py."""
    plad.eval()
    y_true, y_score = [], []
    for x, x_hat in loader:
        x, x_hat = x.to(device), x_hat.to(device)
        p_normal, p_pert, _, _, _ = plad(x, x_hat)
        y_score.extend(p_normal.detach().cpu().numpy().ravel())   # should be low
        y_true.extend(np.zeros(p_normal.shape[0]))
        y_score.extend(p_pert.detach().cpu().numpy().ravel())     # should be high
        y_true.extend(np.ones(p_pert.shape[0]))
    plad.train()
    auc = roc_auc_score(y_true, y_score)
    acc_normal = float(np.mean(np.array(y_score)[np.array(y_true) == 0] < 0.5))
    acc_pert = float(np.mean(np.array(y_score)[np.array(y_true) == 1] >= 0.5))
    return auc, acc_normal, acc_pert


@torch.no_grad()
def export_abnormal(plad, loader, device, out_dir, n_batches=1):
    """Save reconstructed abnormal patches (x_pert) for inspection.
    Each .pkl = {'img': x, 'recon': x_hat, 'abnormal': x_pert} (all H,W,C)."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for b, (x, x_hat) in enumerate(loader):
        x, x_hat = x.to(device), x_hat.to(device)
        _, _, _, _, x_pert = plad(x, x_hat)
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()
        xhat_np = x_hat.permute(0, 2, 3, 1).cpu().numpy()
        xpert_np = x_pert.permute(0, 2, 3, 1).cpu().numpy()
        for j in range(x_np.shape[0]):
            pickle.dump(
                {"img": x_np[j], "recon": xhat_np[j], "abnormal": xpert_np[j]},
                open(os.path.join(out_dir, f"abnormal_{saved}.pkl"), "wb"),
            )
            saved += 1
        if b + 1 >= n_batches:
            break
    print(f"exported {saved} abnormal patches to {out_dir}")




def save_checkpoint(plad, out_dir, tag, epoch, auc=None):
    """Save perturbator+classifier so pseudo-absences can be extracted later.
    Mirrors the old pipeline's model saving."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"plad_{tag}.pth")
    torch.save({
        "perturbator": plad.perturbator.state_dict(),
        "classifier": plad.classifier.state_dict(),
        "epoch": epoch,
        "auc": auc,
    }, path)
    print(f"saved checkpoint -> {path}")


def train(train_folder, val_folder=None, use_vit=True, export_dir=None, output_dir=None):
    device = get_device()
    print(f"training on {device}")
    global lambd

    train_loader = DataLoader(ReconDataset(train_folder), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = None
    if val_folder is not None and os.path.isdir(val_folder):
        val_loader = DataLoader(ReconDataset(val_folder), batch_size=BATCH_SIZE,
                                shuffle=False, drop_last=False)

    plad = PLAD(img_size=32, in_chans=19, device=device, use_vit=use_vit)
    pert_optimizer = torch.optim.Adam(plad.perturbator.parameters())
    clf_optimizer = torch.optim.Adam(plad.classifier.parameters())

    best_auc = 0.0
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (x, x_hat) in enumerate(train_loader):
            x, x_hat = x.to(device), x_hat.to(device)

            pert_optimizer.zero_grad()
            clf_optimizer.zero_grad()

            outputs = plad(x, x_hat)
            loss = L(*outputs[:4])
            loss.backward()

            pert_optimizer.step()
            if i % 200 == 0:
                lambd += 0.001
            clf_optimizer.step()
            epoch_loss += loss.item()

        msg = f"EPOCH {epoch + 1}: loss={epoch_loss / len(train_loader):.4f}"
        if val_loader is not None:
            auc, acc_n, acc_p = evaluate(plad, val_loader, device)
            msg += f" | val_AUC={auc:.4f} acc_normal={acc_n:.2f} acc_pert={acc_p:.2f}"
            if auc > best_auc:
                best_auc = auc
                if output_dir is not None:
                    save_checkpoint(plad, output_dir, 'best', epoch + 1, auc)
        print(msg)

    if val_loader is not None:
        print(f"BEST val AUC: {best_auc:.4f}")
    if output_dir is not None:
        save_checkpoint(plad, output_dir, 'final', EPOCHS, best_auc if val_loader is not None else None)
    if export_dir is not None:
        loader_for_export = val_loader if val_loader is not None else train_loader
        export_abnormal(plad, loader_for_export, device, export_dir, n_batches=1)

    return plad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-folder", required=True,
                    help="TRAIN folder of NORMAL .pkl reconstructions (e.g. .../train/normal)")
    ap.add_argument("--val-folder", type=str, default=None,
                    help="VAL folder of NORMAL .pkl reconstructions (e.g. .../val/normal). "
                         "Enables held-out AUC.")
    ap.add_argument("--conv", action="store_true",
                    help="use the fallback conv classifier instead of the ViT")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="directory to save the trained plad checkpoint (.pth)")
    ap.add_argument("--export-abnormal", type=str, default=None,
                    help="after training, save a batch of abnormal patches here for viewing")
    args = ap.parse_args()
    train(args.recon_folder, val_folder=args.val_folder,
          use_vit=not args.conv, export_dir=args.export_abnormal,
          output_dir=args.output_dir)