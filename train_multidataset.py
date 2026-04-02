"""
train_multidataset.py -- Multi-dataset training for Fed-Breath

Supports training on BIDMC, CapnoBase, or both, with cross-dataset evaluation.

Usage:
    # Train on BIDMC, evaluate on CapnoBase
    python train_multidataset.py --datasets bidmc --eval-on capnobase --epochs 20

    # Joint training (BIDMC + CapnoBase), evaluate on CapnoBase
    python train_multidataset.py --datasets bidmc capnobase --eval-on capnobase --epochs 25

    # Train on BIDMC only (default)
    python train_multidataset.py --datasets bidmc --epochs 20
"""

import os
import argparse
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import BidmcDataset
from capnobase_loader import CapnoBaseDataset
from model import AttentionBiLSTM, rr_to_anomaly_label, NUM_ANOMALY_CLASSES

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BATCH_SIZE     = 8
LEARNING_RATE  = 0.001
LAMBDA_ANOMALY = 0.3      # anomaly loss weight (matches all other modules)
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(name: str):
    """Load a dataset by name."""
    name = name.lower().strip()
    if name == "bidmc":
        return BidmcDataset()
    elif name in ("capnobase", "capno"):
        return CapnoBaseDataset()
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'bidmc' or 'capnobase'.")


def evaluate_on_dataset(model, dataset, batch_size=BATCH_SIZE, prefix=""):
    """
    Evaluate a trained model on a given dataset.
    Returns dict with MAE, RMSE, Bias, and per-sample predictions.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            rr_pred = model(inputs)
            all_preds.extend(rr_pred.squeeze().cpu().tolist())
            all_targets.extend(targets.tolist())

    preds  = np.array(all_preds)
    tgts   = np.array(all_targets)
    errors = preds - tgts

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    sd   = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    tag = f" ({prefix})" if prefix else ""
    print(f"\n  [Eval{tag}] N={len(preds)}")
    print(f"    MAE  = {mae:.4f} BrPM")
    print(f"    RMSE = {rmse:.4f} BrPM")
    print(f"    Bias = {bias:+.4f} BrPM")
    print(f"    95%% LoA = [{loa_lower:.2f}, {loa_upper:.2f}] BrPM")

    return {
        "mae": mae, "rmse": rmse, "bias": bias,
        "sd": sd, "loa_upper": loa_upper, "loa_lower": loa_lower,
        "n": len(preds), "preds": preds, "targets": tgts,
    }


def train(train_dataset, eval_datasets: dict, epochs: int, save_path: str):
    """
    Train model on train_dataset, evaluate on each dataset in eval_datasets.

    Args:
        train_dataset: PyTorch Dataset for training
        eval_datasets: dict of {"name": Dataset} for evaluation after each epoch
        epochs: number of epochs
        save_path: path to save best model checkpoint

    Returns:
        (model, history, best_eval_results)
    """
    # Split training data 80/20
    n = len(train_dataset)
    val_size = int(0.2 * n)
    train_size = n - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size],
                                     generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[Train] Training: {train_size} samples, Validation: {val_size} samples")
    print(f"[Train] Device: {DEVICE}, Epochs: {epochs}")
    print(f"[Train] Loss: MSE(RR) + {LAMBDA_ANOMALY} * CE(anomaly)")

    model = AttentionBiLSTM().to(DEVICE)
    criterion_rr  = nn.MSELoss()
    criterion_ano = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [], "val_mae": [], "val_rmse": [],
    }
    best_mae  = float("inf")
    best_results = {}

    for epoch in range(epochs):
        # -- Train --
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            rr_pred, _, anomaly_logits = model(
                inputs, return_attention=True, return_anomaly=True
            )

            rr_loss = criterion_rr(rr_pred, targets.unsqueeze(1))
            pseudo = torch.tensor(
                [rr_to_anomaly_label(float(t)) for t in targets],
                dtype=torch.long, device=DEVICE
            )
            ano_loss = criterion_ano(anomaly_logits, pseudo)

            loss = rr_loss + LAMBDA_ANOMALY * ano_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # -- Validate on held-out split --
        model.eval()
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                rr_pred = model(inputs)
                all_preds.extend(rr_pred.squeeze().cpu().tolist())
                all_tgts.extend(targets.tolist())

        preds_np = np.array(all_preds)
        tgts_np  = np.array(all_tgts)
        val_mae  = float(np.mean(np.abs(preds_np - tgts_np)))
        val_rmse = float(np.sqrt(np.mean((preds_np - tgts_np) ** 2)))

        n_batches = max(len(train_loader), 1)
        history["train_loss"].append(total_loss / n_batches)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {total_loss/n_batches:.4f} | "
              f"Val MAE: {val_mae:.3f}  RMSE: {val_rmse:.3f} BrPM")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_path)
            print(f"           -> New best MAE {best_mae:.3f} -- saved to {save_path}")

    # -- Cross-dataset evaluation --
    print("\n" + "="*60)
    print("  CROSS-DATASET EVALUATION")
    print("="*60)

    # Reload best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    for ds_name, ds in eval_datasets.items():
        if len(ds) > 0:
            result = evaluate_on_dataset(model, ds, prefix=ds_name)
            best_results[ds_name] = result

    return model, history, best_results


def plot_training_curves(history: dict, save_path: str):
    """Save training loss and validation error curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], color="#378ADD", linewidth=1.8,
                 label="Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss (Multi-Task)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_mae"], color="#1D9E75", linewidth=1.8,
                 label="Val MAE")
    axes[1].plot(history["val_rmse"], color="#534AB7", linewidth=1.8,
                 label="Val RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (BrPM)")
    axes[1].set_title("Validation Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Train] Learning curve saved -> {save_path}")


def save_results_csv(results: dict, path: str):
    """Save cross-dataset evaluation results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "N", "MAE", "RMSE", "Bias", "SD",
                         "LoA_Lower", "LoA_Upper"])
        for name, r in results.items():
            writer.writerow([
                name, r["n"],
                round(r["mae"], 4), round(r["rmse"], 4),
                round(r["bias"], 4), round(r["sd"], 4),
                round(r["loa_lower"], 4), round(r["loa_upper"], 4),
            ])
    print(f"[Train] Results CSV saved -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fed-Breath multi-dataset training"
    )
    parser.add_argument("--datasets", nargs="+", default=["bidmc"],
                        help="Training datasets (bidmc, capnobase)")
    parser.add_argument("--eval-on", nargs="*", default=None,
                        help="Datasets to evaluate on after training "
                             "(default: same as --datasets)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--save-as", default=None,
                        help="Model checkpoint filename "
                             "(default: model_<datasets>.pth)")
    args = parser.parse_args()

    # -- Build training dataset --
    print("="*60)
    print("  FED-BREATH MULTI-DATASET TRAINING")
    print("="*60)

    train_datasets = []
    ds_names = []
    for name in args.datasets:
        ds = get_dataset(name)
        if len(ds) > 0:
            train_datasets.append(ds)
            ds_names.append(name.lower())
            print(f"[OK] {name}: {len(ds)} samples")
        else:
            print(f"[WARN] {name}: 0 samples -- skipping")

    if not train_datasets:
        print("[!!] No training data available. Exiting.")
        return

    if len(train_datasets) == 1:
        train_ds = train_datasets[0]
    else:
        train_ds = ConcatDataset(train_datasets)

    print(f"\n[Train] Total training samples: {len(train_ds)}")

    # -- Build evaluation datasets --
    eval_names = args.eval_on if args.eval_on else args.datasets
    eval_datasets = {}
    for name in eval_names:
        ds = get_dataset(name)
        if len(ds) > 0:
            eval_datasets[name.lower()] = ds

    # -- Determine save path --
    if args.save_as:
        save_path = args.save_as
    else:
        save_path = f"model_{'_'.join(sorted(ds_names))}.pth"

    # -- Train --
    model, history, results = train(train_ds, eval_datasets,
                                     epochs=args.epochs,
                                     save_path=save_path)

    # -- Save outputs --
    plot_training_curves(history, f"training_curve_{'_'.join(sorted(ds_names))}.png")
    save_results_csv(results, f"cross_dataset_results_{'_'.join(sorted(ds_names))}.csv")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Model          : {save_path}")
    print(f"  Train datasets : {', '.join(ds_names)}")
    print(f"  Eval datasets  : {', '.join(eval_datasets.keys())}")
    for name, r in results.items():
        print(f"  {name:12s} MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"Bias={r['bias']:+.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
