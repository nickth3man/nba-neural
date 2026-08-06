"""Training loop with early stopping for the Era Translator DANN."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import (
    BATCH_SIZE,
    CENTROID_ALIGN_WEIGHT,
    ERA_HEAD_LR,
    LAMBDA_GAMMA,
    LAMBDA_MAX,
    LR,
    LR_ALPHA,
    LR_BETA,
    MAX_EPOCHS,
    MOMENTUM,
    NUM_ERAS,
    PATIENCE,
    PATIENCE_WARMUP_EPOCHS,
    SEED,
    WEIGHT_DECAY,
)


def _lambda_at(epoch: int) -> float:
    """Ganin et al. sigmoid schedule: lambda_p = 2/(1+exp(-gamma*p)) - 1, p in [0,1].

    Ramps smoothly 0 -> LAMBDA_MAX over the whole run (arXiv:1505.07818 sec. 5.2.2).
    """
    import math

    p = epoch / max(MAX_EPOCHS - 1, 1)
    return LAMBDA_MAX * (2.0 / (1.0 + math.exp(-LAMBDA_GAMMA * p)) - 1.0)


def _to_tensors(X, y, era, device):
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
        torch.tensor(era, dtype=torch.long, device=device),
    )


def train_model(
    model,
    X_train, y_train, era_train,
    X_val, y_val, era_val,
    device="cpu",
):
    """Train the DANN. Returns (history, best_state_dict).

    Follows Ganin et al. (arXiv:1505.07818): sigmoid lambda schedule; the era head
    trains on era-balanced batches (inverse-frequency WeightedRandomSampler) so
    majority decades cannot dominate it. Balanced batches only - no class-balanced
    CE weights, since Rangwani et al. (ICML 2022) show weakening the adversarial
    loss is harmful. SGD + momentum per Rangwani (smooth task-loss minima stabilize
    adversarial training) with Ganin LR decay.
    """
    torch.manual_seed(SEED)
    model = model.to(device)
    # Encoder + value head: SGD (smooth task-loss minima stabilize adversarial
    # training, Rangwani ICML 2022). Era head: its own Adam at higher LR so the
    # discriminator keeps pace with the encoder - a collapsed discriminator
    # (observed predicting a single decade) produces a degenerate reversal
    # gradient that never strips era from the latent.
    opt = torch.optim.SGD(
        list(model.encoder.parameters()) + list(model.value_head.parameters()),
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    era_opt = torch.optim.Adam(model.era_head.parameters(), lr=ERA_HEAD_LR)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def centroid_align_loss(z, era):
        """Pull each era's latent centroid toward the global centroid (CAT-style,
        arXiv:2407.12782). Directly minimizes era separability in the latent,
        bypassing the discriminator-collapse failure mode of GRL training."""
        z = torch.nn.functional.normalize(z, dim=1)
        global_c = z.mean(0)
        total = 0.0
        for e in range(NUM_ERAS):
            m = era == e
            if m.sum() < 2:
                continue
            c = z[m].mean(0)
            total = total + ((c - global_c) ** 2).sum()
        return total / NUM_ERAS

    Xt, yt, et = _to_tensors(X_train, y_train, era_train, device)
    Xv, yv, ev = _to_tensors(X_val, y_val, era_val, device)
    counts = np.bincount(era_train, minlength=NUM_ERAS).astype(np.float64)
    # Pure inverse-frequency weighting: every era is equally present per batch,
    # so the era head cannot collapse to majority decades.
    invfreq_w = (1.0 / counts) / (1.0 / counts).sum()
    era_sample_w = invfreq_w[era_train]
    sampler = WeightedRandomSampler(
        era_sample_w, num_samples=len(era_train), replacement=True
    )
    loader = DataLoader(
        TensorDataset(Xt, yt, et), batch_size=BATCH_SIZE, sampler=sampler
    )

    history = {"train_loss": [], "val_loss": [], "val_era_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        lam = _lambda_at(epoch)
        # Ganin LR schedule: mu_p = LR / (1 + alpha*p)^beta
        p = epoch / max(MAX_EPOCHS - 1, 1)
        for g in opt.param_groups:
            g["lr"] = LR / (1.0 + LR_ALPHA * p) ** LR_BETA
        model.train()
        total_loss = 0.0
        for xb, yb, eb in loader:
            # Encoder/value step: MSE + GRL-era CE + CAT centroid alignment.
            opt.zero_grad()
            z, vpred, epred = model(xb, lambda_=lam)
            loss = (
                mse(vpred, yb)
                + lam * ce(epred, eb)
                + CENTROID_ALIGN_WEIGHT * centroid_align_loss(z, eb)
            )
            loss.backward()
            opt.step()
            # Era head step (own optimizer; lambda_=0 disables reversal). Kept
            # purely as a monitoring head - its reversal gradient is unreliable
            # once it collapses, so era removal relies on the centroid loss.
            era_opt.zero_grad()
            _, _, epred2 = model(xb, lambda_=0.0)
            ce(epred2, eb).backward()
            era_opt.step()
            total_loss += loss.item() * len(xb)

        # Validation
        model.eval()
        with torch.no_grad():
            _, vpred, epred = model(Xv, lambda_=lam)
            val_loss = mse(vpred, yv).item()
            val_era_acc = (epred.argmax(1) == ev).float().mean().item()

        train_loss = total_loss / len(X_train)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_era_acc"].append(val_era_acc)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            # Early stopping only engages after the lambda schedule has ramped up.
            if epoch >= PATIENCE_WARMUP_EPOCHS:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    break

        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            print(
                f"epoch {epoch:3d} | lam {lam:.2f} | train_loss {train_loss:.4f} "
                f"| val_loss {val_loss:.4f} | val_era_acc {val_era_acc:.4f}"
            )

    model.load_state_dict(best_state)
    return history, best_state
