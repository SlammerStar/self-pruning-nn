"""
main.py
=======
Entry point for the Self-Pruning Neural Network experiment sweep.

Usage
-----
    # Default (reads configs/config.yaml):
    python main.py

    # Override lambdas or epochs from CLI:
    python main.py --lambdas 0.001 0.01 0.1
    python main.py --epochs 20
    python main.py --config configs/config.yaml --no-plots

Flow
----
  1. Load configs/config.yaml
  2. For each λ:  train SelfPruningNetwork → evaluate → save checkpoint + CSV
  3. Print summary table
  4. Generate all matplotlib plots
  5. Write outputs/results_summary.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import yaml

from training.train import run_experiment
from utils.visualization import generate_all_plots

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "experiment.log", mode="w"),
        ],
    )


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flat_cfg(cfg: dict, overrides: dict = None) -> dict:
    """Flatten nested YAML config into one dict for run_experiment()."""
    out = {
        "input_dim":      cfg["model"]["input_dim"],
        "hidden_dims":    cfg["model"]["hidden_dims"],
        "num_classes":    cfg["model"]["num_classes"],
        "dropout_rate":   cfg["model"]["dropout_rate"],
        "epochs":         cfg["training"]["epochs"],
        "batch_size":     cfg["training"]["batch_size"],
        "learning_rate":  cfg["training"]["learning_rate"],
        "weight_decay":   cfg["training"]["weight_decay"],
        "num_workers":    cfg["training"]["num_workers"],
        "log_every":      cfg["training"]["log_every"],
        "data_dir":       cfg["paths"]["data_dir"],
        "checkpoint_dir": cfg["paths"]["checkpoint_dir"],
        "gate_threshold": cfg["evaluation"]["gate_threshold"],
    }
    if overrides:
        out.update({k: v for k, v in overrides.items() if v is not None})
    return out


# ── Output helpers ────────────────────────────────────────────────────────────

def print_table(results: list) -> None:
    print(f"\n{'─'*54}")
    print(f"  {'λ':>8}   {'Test Accuracy':>14}   {'Sparsity %':>12}")
    print(f"{'─'*54}")
    for r in results:
        print(f"  {r['lam']:>8.4f}   {r['accuracy']*100:>13.2f}%   {r['sparsity']*100:>11.2f}%")
    print(f"{'─'*54}\n")


def save_summary_csv(results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Lambda", "Accuracy (%)", "Sparsity (%)"])
        w.writeheader()
        for r in results:
            w.writerow({
                "Lambda":        r["lam"],
                "Accuracy (%)":  round(r["accuracy"] * 100, 2),
                "Sparsity (%)":  round(r["sparsity"] * 100, 2),
            })
    logging.getLogger(__name__).info(f"Summary CSV → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Pruning NN experiment sweep")
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--lambdas",   nargs="+", type=float, default=None)
    p.add_argument("--epochs",    type=int,   default=None)
    p.add_argument("--no-plots",  action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    setup_logging(cfg["paths"]["logs_dir"])
    log    = logging.getLogger(__name__)

    fc      = flat_cfg(cfg, overrides={"epochs": args.epochs})
    lambdas = args.lambdas or cfg["lambdas"]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon GPU
    else:
        device = torch.device("cpu")

    log.info(f"Device  : {device}")
    log.info(f"Lambdas : {lambdas}")
    log.info(f"Epochs  : {fc['epochs']}")
    log.info(f"Network : {fc['hidden_dims']}")

    results = []
    for lam in lambdas:
        r = run_experiment(lam=lam, cfg=fc, device=device)
        results.append(r)

    print_table(results)
    save_summary_csv(results, cfg["paths"]["results_csv"])

    if not args.no_plots:
        generate_all_plots(results, Path(cfg["paths"]["plots_dir"]))

    log.info("All experiments complete.")


if __name__ == "__main__":
    main()
