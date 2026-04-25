"""
training/train.py
=================
Full training + evaluation pipeline for the Self-Pruning Neural Network.

Usage (called from main.py):
    result = run_experiment(lam=0.01, cfg=flat_cfg, device=device)
"""

import csv
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models.network import SelfPruningNetwork
from training.loss import SparsityRegularisedLoss

logger = logging.getLogger(__name__)


# ── Data ─────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    data_dir:    str = "./data",
    batch_size:  int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 train and test DataLoaders."""

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(
        test_ds,  batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    logger.info(f"CIFAR-10 — train: {len(train_ds):,}  test: {len(test_ds):,}")
    return train_loader, test_loader


# ── Single epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     SelfPruningNetwork,
    loader:    DataLoader,
    criterion: SparsityRegularisedLoss,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> Dict[str, float]:
    model.train()
    total_sum = ce_sum = sp_sum = n = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        total, ce, sp = criterion(logits, labels, model)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_sum += total.item(); ce_sum += ce.item(); sp_sum += sp.item(); n += 1

    return {"total_loss": total_sum/n, "ce_loss": ce_sum/n, "sp_loss": sp_sum/n}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:  SelfPruningNetwork,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        preds    = model(images.to(device)).argmax(dim=1)
        correct += (preds == labels.to(device)).sum().item()
        total   += labels.size(0)
    return {
        "accuracy": correct / total,
        "sparsity": model.sparsity(),
    }


# ── Full experiment ───────────────────────────────────────────────────────────

def run_experiment(
    lam:    float,
    cfg:    Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train one SelfPruningNetwork for a given λ.

    Parameters
    ----------
    lam    : sparsity coefficient
    cfg    : flat config dict (keys: epochs, batch_size, learning_rate,
             weight_decay, hidden_dims, dropout_rate, data_dir,
             checkpoint_dir, num_workers, log_every)
    device : torch device

    Returns
    -------
    dict with: lam, accuracy, sparsity, gates (np.ndarray), history (list)
    """
    logger.info(f"\n{'='*56}\n  λ = {lam}  |  {cfg['epochs']} epochs\n{'='*56}")

    model = SelfPruningNetwork(
        input_dim    = cfg.get("input_dim",    3072),
        hidden_dims  = cfg.get("hidden_dims",  [1024, 512, 256]),
        num_classes  = cfg.get("num_classes",  10),
        dropout_rate = cfg.get("dropout_rate", 0.3),
    ).to(device)

    criterion = SparsityRegularisedLoss(lam=lam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    train_loader, test_loader = get_cifar10_loaders(
        data_dir    = cfg["data_dir"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg.get("num_workers", 2),
    )

    # ── CSV: one row per epoch ─────────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir.parent / f"lambda_{lam}_history.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "total_loss", "ce_loss", "sp_loss",
        "val_accuracy", "val_sparsity", "elapsed_s"])
    writer.writeheader()

    best_acc, best_state, history = 0.0, None, []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m   = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = {**train_m,
               "epoch": epoch,
               "val_accuracy": val_m["accuracy"],
               "val_sparsity": val_m["sparsity"],
               "elapsed_s":    round(elapsed, 2)}
        writer.writerow(row); csv_file.flush()
        history.append(row)

        if val_m["accuracy"] > best_acc:
            best_acc   = val_m["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        log_every = cfg.get("log_every", 5)
        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                f"  ep {epoch:3d}/{cfg['epochs']}  "
                f"loss={train_m['total_loss']:.4f} "
                f"(ce={train_m['ce_loss']:.4f} sp={train_m['sp_loss']:.4f})  "
                f"acc={val_m['accuracy']:.4f}  "
                f"sparsity={val_m['sparsity']:.4f}  "
                f"{elapsed:.1f}s"
            )

    csv_file.close()

    # Reload best
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final = evaluate(model, test_loader, device)
    gates = model.all_gates().cpu().numpy()
    counts = model.param_counts()

    torch.save(best_state, ckpt_dir / f"best_lambda_{lam}.pt")

    logger.info(
        f"\n  FINAL λ={lam}  acc={final['accuracy']:.4f}  "
        f"sparsity={final['sparsity']:.4f}  {counts}"
    )

    del model; gc.collect()

    return {
        "lam":      lam,
        "accuracy": final["accuracy"],
        "sparsity": final["sparsity"],
        "gates":    gates,
        "history":  history,
        "counts":   counts,
    }
