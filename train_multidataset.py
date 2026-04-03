"""
train_multidataset.py — Multi-Dataset Training for Fed-Breath

Trains AttentionBiLSTM on any combination of BIDMC, CapnoBase, and Apnea-ECG.
Produces a checkpoint ready for evaluation and FL deployment.

Usage examples:
    # BIDMC only (baseline, same as train_centralized.py)
    python train_multidataset.py --datasets bidmc

    # BIDMC + CapnoBase (recommended — standard dual-benchmark)
    python train_multidataset.py --datasets bidmc capnobase

    # All three (maximum coverage)
    python train_multidataset.py --datasets bidmc capnobase apnea

    # Evaluate on CapnoBase after training on BIDMC (cross-dataset generalisation)
    python train_multidataset.py --datasets bidmc --eval-on capnobase

Options:
    --epochs     N        training epochs (default: 20)
    --batch      N        batch size (default: 8)
    --lr         F        learning rate (default: 0.001)
    --lambda-ano F        anomaly loss weight (default: 0.3)
    --out        path     output model path (default: auto-named)
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import (
    BidmcDataset, CapnoBaseDataset, ApneaECGDataset,
    CombinedDataset, available_datasets
)
from model import AttentionBiLSTM, rr_to_anomaly_label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Per-dataset evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_on_dataset(model, dataset, batch_size=8, label="") -> dict:
    """
    Runs evaluation on a single dataset.
    Returns dict with MAE, RMSE, bias, within2, within5.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds_all, tgts_all = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            inputs = inputs.to(DEVICE)
            rr_pred, _ = model(inputs, return_attention=True)
            preds_all.extend(rr_pred.squeeze().cpu().tolist())
            tgts_all.extend(targets.tolist())

    preds   = np.array(preds_all)
    targets = np.array(tgts_all)
    errors  = preds - targets

    mae     = float(np.mean(np.abs(errors)))
    rmse    = float(np.sqrt(np.mean(errors ** 2)))
    bias    = float(np.mean(errors))
    within2 = float(np.mean(np.abs(errors) <= 2) * 100)
    within5 = float(np.mean(np.abs(errors) <= 5) * 100)

    tag = f"[{label}]" if label else "[Eval]"
    print(f"{tag} MAE={mae:.3f}  RMSE={rmse:.3f}  "
          f"Bias={bias:+.3f}  Within2={within2:.1f}%  Within5={within5:.1f}%")
    return {"mae": mae, "rmse": rmse, "bias": bias,
            "within2": within2, "within5": within5, "n": len(preds)}


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────
def train(args):
    print(f"\nFed-Breath Multi-Dataset Training")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  LR         : {args.lr}")
    print(f"  Lambda-ano : {args.lambda_ano}")

    # ── Check dataset availability ────────────────────────────
    status = available_datasets()
    print("\nDataset availability:")
    for name, info in status.items():
        state = "READY" if info["ready"] else "missing"
        print(f"  {name:<12}: {state}  ({info['files']} files)")

    # ── Load datasets ─────────────────────────────────────────
    available = [s for s in args.datasets if status.get(s, {}).get("ready")]
    missing   = [s for s in args.datasets if s not in available]
    if missing:
        print(f"\n[WARN] These datasets not found and will be skipped: {missing}")
        print(f"       See dataset.py docstrings for download instructions.")
    if not available:
        print("No datasets available. Exiting.")
        return

    print(f"\nLoading: {available}")
    if len(available) == 1:
        src = available[0]
        ds_map = {"bidmc": BidmcDataset, "capnobase": CapnoBaseDataset}
        full_ds = ds_map[src]()
    else:
        full_ds = CombinedDataset(available)

    total     = len(full_ds)
    val_size  = max(1, int(0.2 * total))
    train_size = total - val_size

    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    print(f"Train: {train_size}  Val: {val_size}")

    # ── Load separate eval datasets ───────────────────────────
    eval_loaders = {}
    if args.eval_on:
        for src in args.eval_on:
            if status.get(src, {}).get("ready"):
                try:
                    ds_map = {
                        "bidmc":    BidmcDataset,
                        "capnobase":CapnoBaseDataset,
                    }
                    eval_loaders[src] = DataLoader(
                        ds_map[src](), batch_size=args.batch, shuffle=False
                    )
                    print(f"[Eval-on] {src}: {len(eval_loaders[src].dataset)} samples")
                except Exception as e:
                    print(f"[Eval-on] Could not load {src}: {e}")
            else:
                print(f"[Eval-on] {src} not available — skipping")

    # ── Model ─────────────────────────────────────────────────
    model        = AttentionBiLSTM().to(DEVICE)
    criterion_rr = nn.MSELoss()
    criterion_ano= nn.CrossEntropyLoss()
    optimizer    = optim.Adam(model.parameters(), lr=args.lr)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # ── Output paths ──────────────────────────────────────────
    tag = "_".join(sorted(available))
    out_model   = args.out or f"model_{tag}.pth"
    out_csv     = f"train_log_{tag}.csv"
    out_plot    = f"training_curve_{tag}.png"

    # ── Training ──────────────────────────────────────────────
    history  = {"train_loss": [], "val_mae": [], "val_rmse": []}
    best_mae = float("inf")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Epoch","TrainLoss","RRLoss","AnoLoss","ValMAE","ValRMSE"])

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # ── Train epoch ──────────────────────────────────────
        model.train()
        total_loss = rr_loss_sum = ano_loss_sum = 0.0
        n_batches  = 0

        for batch in train_loader:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            rr_pred, _, ano_logits = model(
                inputs, return_attention=True, return_anomaly=True
            )
            rr_loss  = criterion_rr(rr_pred, targets.unsqueeze(1))
            pseudo   = torch.tensor(
                [rr_to_anomaly_label(float(t)) for t in targets],
                dtype=torch.long, device=DEVICE
            )
            ano_loss = criterion_ano(ano_logits, pseudo)
            loss     = rr_loss + args.lambda_ano * ano_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss   += loss.item()
            rr_loss_sum  += rr_loss.item()
            ano_loss_sum += ano_loss.item()
            n_batches    += 1

        avg_loss = total_loss   / max(n_batches, 1)
        avg_rr   = rr_loss_sum  / max(n_batches, 1)
        avg_ano  = ano_loss_sum / max(n_batches, 1)

        # ── Val epoch ────────────────────────────────────────
        res = evaluate_on_dataset(model, val_ds, args.batch, label=f"E{epoch+1:02d} val")

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(res["mae"])
        history["val_rmse"].append(res["rmse"])

        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, round(avg_loss,4), round(avg_rr,4), round(avg_ano,4),
                round(res["mae"],4), round(res["rmse"],4)
            ])

        print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss={avg_loss:.4f} (RR={avg_rr:.4f} Ano={avg_ano:.4f}) | "
              f"ValMAE={res['mae']:.3f}  ValRMSE={res['rmse']:.3f}")

        # ── LR scheduler ─────────────────────────────────────
        scheduler.step(res["mae"])

        # ── Save best ─────────────────────────────────────────
        if res["mae"] < best_mae:
            best_mae = res["mae"]
            torch.save(model.state_dict(), out_model)
            print(f"           -> Best MAE {best_mae:.3f} — saved to {out_model}")

    # ── Cross-dataset evaluation ──────────────────────────────
    if eval_loaders:
        print(f"\nCross-dataset evaluation (model: {out_model}):")
        model.load_state_dict(torch.load(out_model, map_location=DEVICE))
        for src, ldr in eval_loaders.items():
            evaluate_on_dataset(model, ldr.dataset, args.batch, label=src.upper())

    # ── Learning curve ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], color="#378ADD", linewidth=1.8, label="Train loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_mae"],  color="#1D9E75", linewidth=1.8, label="Val MAE")
    axes[1].plot(history["val_rmse"], color="#534AB7", linewidth=1.8, label="Val RMSE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Error (BrPM)")
    axes[1].set_title("Validation error")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Datasets: {', '.join(available)} — Best MAE: {best_mae:.3f} BrPM",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTraining complete.")
    print(f"  Best Val MAE : {best_mae:.4f} BrPM")
    print(f"  Model saved  : {out_model}")
    print(f"  Training log : {out_csv}")
    print(f"  Learning plot: {out_plot}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fed-Breath multi-dataset training")
    parser.add_argument("--datasets",   nargs="+",
                        default=["bidmc"],
                        choices=["bidmc","capnobase","apnea"],
                        help="Datasets to train on (default: bidmc)")
    parser.add_argument("--eval-on",    nargs="*",
                        default=[],
                        choices=["bidmc","capnobase"],
                        help="Additional datasets to evaluate on after training")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--lambda-ano", type=float, default=0.3,
                        dest="lambda_ano")
    parser.add_argument("--out",        type=str,   default=None,
                        help="Output model path (default: model_<datasets>.pth)")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()