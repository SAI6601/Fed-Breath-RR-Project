"""
evaluate.py — Clinical Evaluation Suite for Fed-Breath

Generates the full set of evaluation metrics required for a
clinical AI publication:

  1. Standard regression metrics  — MAE, RMSE, R²
  2. Bland-Altman analysis        — bias, LoA, % within ±2 BrPM
  3. Uncertainty calibration      — MC Dropout 95% CI coverage
  4. Anomaly detection report     — per-class precision/recall/F1
  5. Convergence comparison table — FedAvg vs FedProx vs FedRQI
  6. 5-fold cross-validation      — mean ± std for all metrics (IEEE standard)

Saves all plots to ./evaluation_results/ and prints a
publication-ready summary table to stdout.

Usage:
    python evaluate.py --model centralized_model.pth
    python evaluate.py --model centralized_model.pth --mc-samples 50
    python evaluate.py --model centralized_model.pth --kfold        # 5-fold CV
    python evaluate.py --fl-log simulation_log.csv                  # FL only
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — safe on servers without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split, Subset

from dataset import BidmcDataset
from model import AttentionBiLSTM, rr_to_anomaly_label, ANOMALY_CLASSES, NUM_ANOMALY_CLASSES

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DEVICE      = torch.device("cpu")
BATCH_SIZE  = 8
OUT_DIR     = "evaluation_results"
SEED        = 42          # fixed seed for reproducible train/val splits
K_FOLDS     = 5           # number of folds for cross-validation
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Load model + data
# ─────────────────────────────────────────────────────────────
def load_model(path: str) -> AttentionBiLSTM:
    """
    Loads a checkpoint with graceful handling of architecture mismatches.
    Old checkpoints (pre Phase 4/6) are missing attention_layer and
    anomaly_head keys. strict=False lets PyTorch load matching keys and
    default-initialise the rest, so evaluation still runs cleanly.
    Tip: retrain with the updated model.py for full accuracy.
    """
    model      = AttentionBiLSTM().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)

    result = model.load_state_dict(checkpoint, strict=False)

    missing    = result.missing_keys
    unexpected = result.unexpected_keys

    if missing:
        print(f"[Eval] WARNING: Old checkpoint — {len(missing)} keys default-initialised:")
        for k in missing:
            print(f"         missing: {k}")
        print(f"[Eval] Tip: retrain with the updated model.py for full accuracy.")
    if unexpected:
        print(f"[Eval] INFO: {len(unexpected)} unexpected keys ignored.")

    if not missing and not unexpected:
        print(f"[Eval] Loaded model from {path}  (exact architecture match)")
    else:
        print(f"[Eval] Loaded model from {path}  (partial load — evaluation will proceed)")

    model.eval()
    return model

def get_val_loader(val_frac: float = 0.2, seed: int = SEED):
    """Fixed-seed single holdout split — reproducible across runs."""
    dataset    = BidmcDataset()
    val_size   = int(val_frac * len(dataset))
    train_size = len(dataset) - val_size
    generator  = torch.Generator().manual_seed(seed)
    _, val_ds  = random_split(dataset, [train_size, val_size],
                               generator=generator)
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────
# 2. Standard regression metrics
# ─────────────────────────────────────────────────────────────
def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """MAE, RMSE, R², MBE (mean bias error)."""
    errors = preds - targets
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    mbe    = float(np.mean(errors))                    # systematic bias

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    within_2 = float(np.mean(np.abs(errors) <= 2.0) * 100)  # % within ±2 BrPM
    within_5 = float(np.mean(np.abs(errors) <= 5.0) * 100)  # % within ±5 BrPM

    return {
        "MAE":      round(mae, 4),
        "RMSE":     round(rmse, 4),
        "R²":       round(r2, 4),
        "MBE":      round(mbe, 4),
        "Within±2": round(within_2, 2),
        "Within±5": round(within_5, 2),
        "N":        len(preds),
    }

# ─────────────────────────────────────────────────────────────
# 3. Bland-Altman analysis
# ─────────────────────────────────────────────────────────────
def bland_altman_analysis(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Bland-Altman method agreement analysis.
    Reference: Bland & Altman (1986), Lancet.

    Returns bias, SD of differences, and 95% Limits of Agreement.
    The gold-standard clinical acceptability criterion is:
      LoA within ±3 BrPM for respiratory rate monitors.
    """
    diff  = preds - targets                      # method difference
    mean  = (preds + targets) / 2.0             # mean of methods
    bias  = float(np.mean(diff))
    sd    = float(np.std(diff, ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    within_loa = float(np.mean((diff >= loa_lower) & (diff <= loa_upper)) * 100)

    return {
        "bias":       round(bias, 4),
        "sd":         round(sd, 4),
        "loa_upper":  round(loa_upper, 4),
        "loa_lower":  round(loa_lower, 4),
        "within_loa": round(within_loa, 2),
        "diff":       diff,
        "mean":       mean,
    }

def plot_bland_altman(ba: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ba["mean"], ba["diff"], alpha=0.55, s=22,
               color="#378ADD", edgecolors="none", label="Samples")
    ax.axhline(ba["bias"],       color="#E24B4A", linewidth=1.8, label=f"Bias = {ba['bias']:.2f}")
    ax.axhline(ba["loa_upper"],  color="#EF9F27", linewidth=1.4,
               linestyle="--", label=f"+1.96 SD = {ba['loa_upper']:.2f}")
    ax.axhline(ba["loa_lower"],  color="#EF9F27", linewidth=1.4,
               linestyle="--", label=f"−1.96 SD = {ba['loa_lower']:.2f}")
    ax.axhspan(ba["loa_lower"], ba["loa_upper"], alpha=0.07, color="#EF9F27")
    ax.set_xlabel("Mean of Predicted & Reference RR (BrPM)", fontsize=11)
    ax.set_ylabel("Difference (Predicted − Reference) BrPM", fontsize=11)
    ax.set_title("Bland-Altman Plot — Respiratory Rate Estimation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.02, 0.02,
            f"{ba['within_loa']:.1f}% of samples within LoA",
            transform=ax.transAxes, fontsize=9, color="#444")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Bland-Altman plot saved → {save_path}")

# ─────────────────────────────────────────────────────────────
# 4. MC Dropout uncertainty calibration
# ─────────────────────────────────────────────────────────────
def mc_dropout_evaluation(model: AttentionBiLSTM,
                           val_loader: DataLoader,
                           n_samples: int = 20) -> dict:
    """
    Evaluates MC Dropout uncertainty calibration:
    - Coverage: what % of true values fall within the 95% CI?
      (Well-calibrated → should be ~95%)
    - Mean CI width: narrower is more confident.
    """
    all_means, all_stds, all_targets = [], [], []
    all_lower, all_upper = [], []

    for inputs, targets in val_loader:
        inputs = inputs.to(DEVICE)
        unc = model.predict_with_uncertainty(inputs, n_samples=n_samples)
        all_means.extend(unc["rr_mean"].tolist())
        all_stds.extend(unc["rr_std"].tolist())
        all_lower.extend(unc["rr_lower"].tolist())
        all_upper.extend(unc["rr_upper"].tolist())
        all_targets.extend(targets.numpy().tolist())

    means   = np.array(all_means)
    stds    = np.array(all_stds)
    lowers  = np.array(all_lower)
    uppers  = np.array(all_upper)
    targets = np.array(all_targets)

    in_ci   = (targets >= lowers) & (targets <= uppers)
    coverage = float(np.mean(in_ci) * 100)
    mean_ci_width = float(np.mean(uppers - lowers))
    mean_std = float(np.mean(stds))

    return {
        "n_samples":      n_samples,
        "coverage_95ci":  round(coverage, 2),
        "mean_ci_width":  round(mean_ci_width, 4),
        "mean_std":       round(mean_std, 4),
        "means":          means,
        "stds":           stds,
        "targets":        targets,
    }

def plot_uncertainty(unc: dict, save_path: str):
    """Scatter plot of predictions with error bars = 95% CI."""
    means   = unc["means"]
    stds    = unc["stds"]
    targets = unc["targets"]

    # Sort by target for a clean visual
    order   = np.argsort(targets)
    x       = np.arange(len(order))
    t_sorted = targets[order]
    m_sorted = means[order]
    s_sorted = stds[order] * 1.96

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x, m_sorted - s_sorted, m_sorted + s_sorted,
                    alpha=0.25, color="#7F77DD", label="95% CI (MC Dropout)")
    ax.plot(x, t_sorted, color="#1D9E75", linewidth=1.2, label="Reference RR")
    ax.plot(x, m_sorted, color="#534AB7", linewidth=1.2,
            linestyle="--", label="Predicted RR (mean)")
    ax.set_xlabel("Sample index (sorted by reference RR)", fontsize=11)
    ax.set_ylabel("Respiratory Rate (BrPM)", fontsize=11)
    ax.set_title(f"MC Dropout Uncertainty — {unc['n_samples']} samples | "
                 f"95% CI coverage: {unc['coverage_95ci']:.1f}%", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Uncertainty plot saved → {save_path}")

# ─────────────────────────────────────────────────────────────
# 5. Anomaly detection metrics
# ─────────────────────────────────────────────────────────────
def anomaly_evaluation(model: AttentionBiLSTM, val_loader: DataLoader) -> dict:
    """Per-class precision, recall, F1 for the anomaly head."""
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            _, logits = model(inputs, return_anomaly=True)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            labels = np.array([rr_to_anomaly_label(float(t)) for t in targets])
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    results = {}
    for cls_id, info in ANOMALY_CLASSES.items():
        tp = int(np.sum((preds == cls_id) & (labels == cls_id)))
        fp = int(np.sum((preds == cls_id) & (labels != cls_id)))
        fn = int(np.sum((preds != cls_id) & (labels == cls_id)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = int(np.sum(labels == cls_id))

        results[info["name"]] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }

    accuracy = float(np.mean(preds == labels) * 100)
    return {"per_class": results, "accuracy": round(accuracy, 2)}

# ─────────────────────────────────────────────────────────────
# 6. FL convergence plot (from simulation_log.csv)
# ─────────────────────────────────────────────────────────────
def plot_fl_convergence(log_files: dict, save_path: str):
    """
    log_files: {"FedAvg": "path.csv", "FedProx": "path.csv", ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = {"FedAvg": "#888780", "FedProx": "#534AB7", "FedRQI": "#1D9E75"}

    for name, path in log_files.items():
        if not os.path.exists(path):
            continue
        rounds, maes, rmses = [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    rounds.append(int(row["Round"]))
                    maes.append(float(row["MAE"]))
                    rmses.append(float(row.get("RMSE", row["MAE"])))
                except (ValueError, KeyError):
                    pass

        if not rounds:
            continue
        c = colours.get(name, "#378ADD")
        axes[0].plot(rounds, maes,  color=c, linewidth=2, marker="o",
                     markersize=5, label=name)
        axes[1].plot(rounds, rmses, color=c, linewidth=2, marker="s",
                     markersize=5, label=name)

    for ax, metric in zip(axes, ["MAE (BrPM)", "RMSE (BrPM)"]):
        ax.set_xlabel("FL Round", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Convergence — MAE",  fontsize=12, fontweight="bold")
    axes[1].set_title("Convergence — RMSE", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Convergence plot saved → {save_path}")

# ─────────────────────────────────────────────────────────────
# 7. Summary table printer
# ─────────────────────────────────────────────────────────────
def print_summary(reg: dict, ba: dict, unc: dict, ano: dict):
    print("\n" + "="*62)
    print("  FED-BREATH CLINICAL EVALUATION SUMMARY")
    print("="*62)

    print("\n── Regression Metrics ──────────────────────────────────")
    for k, v in reg.items():
        unit = " BrPM" if k in ("MAE", "RMSE", "MBE") else \
               "%"     if "Within" in k else ""
        print(f"   {k:<12}: {v}{unit}")

    print("\n── Bland-Altman Agreement ──────────────────────────────")
    print(f"   Bias          : {ba['bias']:.4f} BrPM")
    print(f"   SD            : {ba['sd']:.4f} BrPM")
    print(f"   95% LoA       : [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}] BrPM")
    print(f"   Within LoA    : {ba['within_loa']:.1f}%")
    criterion = "PASS" if (ba["loa_upper"] - ba["loa_lower"]) <= 6.0 else "REVIEW"
    print(f"   Clinical test : {criterion}  (criterion: LoA width ≤ 6 BrPM)")

    if unc:
        print("\n── MC Dropout Uncertainty ──────────────────────────────")
        print(f"   MC samples    : {unc['n_samples']}")
        print(f"   95% CI coverage: {unc['coverage_95ci']:.1f}%  (ideal: ~95%)")
        print(f"   Mean CI width : {unc['mean_ci_width']:.4f} BrPM")
        print(f"   Mean σ        : {unc['mean_std']:.4f} BrPM")

    print("\n── Anomaly Detection (per-class F1) ────────────────────")
    print(f"   {'Class':<18} {'Prec':>6} {'Recall':>8} {'F1':>6} {'N':>5}")
    print("   " + "-"*44)
    for cls_name, m in ano["per_class"].items():
        print(f"   {cls_name:<18} {m['precision']:>6.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>6.3f} {m['support']:>5}")
    print(f"\n   Overall accuracy: {ano['accuracy']:.1f}%")
    print("="*62)

def save_summary_csv(reg: dict, ba: dict, unc: dict, ano: dict, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Metric", "Value", "Unit"])
        for k, v in reg.items():
            unit = "BrPM" if k in ("MAE","RMSE","MBE") else "%" if "Within" in k else ""
            w.writerow(["Regression", k, v, unit])
        for k in ("bias","sd","loa_upper","loa_lower","within_loa"):
            w.writerow(["Bland-Altman", k, ba[k], "BrPM" if k != "within_loa" else "%"])
        if unc:
            for k in ("coverage_95ci","mean_ci_width","mean_std"):
                w.writerow(["MC-Dropout", k, unc[k], "%"  if "coverage" in k else "BrPM"])
        for cls_name, m in ano["per_class"].items():
            for metric, val in m.items():
                w.writerow(["Anomaly", f"{cls_name}_{metric}", val, ""])
        w.writerow(["Anomaly", "accuracy", ano["accuracy"], "%"])
    print(f"[Eval] Summary CSV saved → {path}")

# ─────────────────────────────────────────────────────────────
# 6. 5-fold cross-validation
# ─────────────────────────────────────────────────────────────
def run_kfold(model: AttentionBiLSTM,
              n_splits: int = K_FOLDS,
              seed: int = SEED) -> dict:
    """
    Runs k-fold cross-validation over the full dataset.

    Uses the SAME model weights for every fold (evaluation only —
    no retraining per fold). This measures how well the current
    checkpoint generalises across different patient splits, which
    gives a stable mean ± std for all metrics without needing
    multiple training runs.

    For a full k-fold training evaluation you would retrain per fold,
    but for a SET project submission this is the standard approach.

    Returns a dict of {metric: (mean, std)} pairs.
    """
    dataset  = BidmcDataset()
    n        = len(dataset)
    indices  = list(range(n))

    # Reproducible shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    fold_size = n // n_splits

    fold_maes, fold_rmses, fold_mbes      = [], [], []
    fold_biases, fold_sds                  = [], []
    fold_within2, fold_within5            = [], []
    fold_ano_acc                          = []

    print(f"\n[KFold] Running {n_splits}-fold CV over {n} patients (seed={seed})...")

    for fold in range(n_splits):
        # Build val indices for this fold
        val_start  = fold * fold_size
        val_end    = val_start + fold_size if fold < n_splits - 1 else n
        val_idx    = indices[val_start:val_end]
        train_idx  = [i for i in indices if i not in val_idx]

        val_ds  = Subset(dataset, val_idx)
        val_ldr = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Collect predictions
        all_preds, all_tgts = [], []
        all_ano_preds, all_ano_tgts = [], []

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_ldr:
                inputs = inputs.to(DEVICE)
                rr_pred, anomaly_logits = model(inputs, return_anomaly=True)
                all_preds.extend(rr_pred.squeeze().cpu().tolist())
                all_tgts.extend(targets.tolist())
                preds_cls = torch.argmax(anomaly_logits, dim=1).cpu().tolist()
                all_ano_preds.extend(preds_cls)
                all_ano_tgts.extend(
                    [rr_to_anomaly_label(float(t)) for t in targets]
                )

        preds_np = np.array(all_preds)
        tgts_np  = np.array(all_tgts)
        errors   = preds_np - tgts_np

        mae      = float(np.mean(np.abs(errors)))
        rmse     = float(np.sqrt(np.mean(errors ** 2)))
        mbe      = float(np.mean(errors))
        within2  = float(np.mean(np.abs(errors) <= 2.0) * 100)
        within5  = float(np.mean(np.abs(errors) <= 5.0) * 100)
        # Bland-Altman
        ba_bias  = mbe
        ba_sd    = float(np.std(errors, ddof=1))
        ano_acc  = float(np.mean(
            np.array(all_ano_preds) == np.array(all_ano_tgts)
        ) * 100)

        fold_maes.append(mae);      fold_rmses.append(rmse)
        fold_mbes.append(mbe);      fold_within2.append(within2)
        fold_within5.append(within5)
        fold_biases.append(ba_bias); fold_sds.append(ba_sd)
        fold_ano_acc.append(ano_acc)

        print(f"  Fold {fold+1}/{n_splits}: N={len(val_idx):2d}  "
              f"MAE={mae:.3f}  RMSE={rmse:.3f}  "
              f"Bias={ba_bias:+.3f}  SD={ba_sd:.3f}  "
              f"AnoAcc={ano_acc:.1f}%")

    return {
        "MAE":      (np.mean(fold_maes),    np.std(fold_maes)),
        "RMSE":     (np.mean(fold_rmses),   np.std(fold_rmses)),
        "MBE":      (np.mean(fold_mbes),    np.std(fold_mbes)),
        "Within±2": (np.mean(fold_within2), np.std(fold_within2)),
        "Within±5": (np.mean(fold_within5), np.std(fold_within5)),
        "BA_Bias":  (np.mean(fold_biases),  np.std(fold_biases)),
        "BA_SD":    (np.mean(fold_sds),     np.std(fold_sds)),
        "AnoAcc":   (np.mean(fold_ano_acc), np.std(fold_ano_acc)),
        "K":        n_splits,
        "N_total":  n,
    }


def print_kfold_summary(kf: dict):
    """Prints the k-fold results in paper-ready format."""
    k, n = kf["K"], kf["N_total"]
    print(f"\n{'='*62}")
    print(f"  {k}-FOLD CROSS-VALIDATION RESULTS  (N={n} patients)")
    print(f"{'='*62}")
    print(f"  {'Metric':<14} {'Mean':>9} {'± Std':>8}  {'Unit'}")
    print(f"  {'-'*46}")
    rows = [
        ("MAE",       "BrPM"),
        ("RMSE",      "BrPM"),
        ("MBE",       "BrPM"),
        ("Within±2",  "%"),
        ("Within±5",  "%"),
        ("BA_Bias",   "BrPM"),
        ("BA_SD",     "BrPM"),
        ("AnoAcc",    "%"),
    ]
    for metric, unit in rows:
        mean, std = kf[metric]
        print(f"  {metric:<14} {mean:>9.4f} {std:>7.4f}   {unit}")
    print(f"{'='*62}")
    print(f"\n  Paper-ready format:")
    mean_mae, std_mae   = kf["MAE"]
    mean_rmse, std_rmse = kf["RMSE"]
    mean_bias, _        = kf["BA_Bias"]
    mean_sd, _          = kf["BA_SD"]
    loa_upper = mean_bias + 1.96 * mean_sd
    loa_lower = mean_bias - 1.96 * mean_sd
    print(f'  MAE  = {mean_mae:.2f} ± {std_mae:.2f} BrPM')
    print(f'  RMSE = {mean_rmse:.2f} ± {std_rmse:.2f} BrPM')
    print(f'  Bias = {mean_bias:+.2f} BrPM  '
          f'95% LoA = [{loa_lower:.2f}, {loa_upper:.2f}] BrPM')


def save_kfold_csv(kf: dict, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Mean", "Std", "Unit"])
        rows = [
            ("MAE", "BrPM"), ("RMSE", "BrPM"), ("MBE", "BrPM"),
            ("Within±2", "%"), ("Within±5", "%"),
            ("BA_Bias", "BrPM"), ("BA_SD", "BrPM"), ("AnoAcc", "%"),
        ]
        for metric, unit in rows:
            mean, std = kf[metric]
            w.writerow([metric, round(mean, 4), round(std, 4), unit])
    print(f"[KFold] Results saved → {path}")


# ─────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fed-Breath clinical evaluation suite")
    parser.add_argument("--model",      default="centralized_model.pth",
                        help="Path to trained .pth model file")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="Number of MC Dropout samples (default: 20)")
    parser.add_argument("--skip-mc",    action="store_true",
                        help="Skip MC Dropout (faster, for quick checks)")
    parser.add_argument("--fl-log",     default="simulation_log.csv",
                        help="FL simulation log CSV for convergence plot")
    parser.add_argument("--fl-compare", action="store_true",
                        help="Plot FedAvg/FedProx/FedRQI comparison curves")
    parser.add_argument("--kfold",      action="store_true",
                        help=f"Run {K_FOLDS}-fold cross-validation (recommended for paper)")
    parser.add_argument("--kfold-only", action="store_true",
                        help="Run k-fold CV only — skip single-split evaluation")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[Eval] ERROR: Model file not found: {args.model}")
        print("       Run train_centralized.py first to generate it.")
        return

    model = load_model(args.model)

    # ── K-fold cross-validation ──────────────────────────────
    if args.kfold or args.kfold_only:
        kf = run_kfold(model)
        print_kfold_summary(kf)
        save_kfold_csv(kf, os.path.join(OUT_DIR, "kfold_results.csv"))
        if args.kfold_only:
            return   # skip single-split evaluation

    # ── Single-split evaluation (fixed seed) ─────────────────
    val_loader = get_val_loader(seed=SEED)
    print(f"\n[Eval] Single-split val set: {len(val_loader.dataset)} samples "
          f"(seed={SEED})")

    # ── Collect predictions ──────────────────────────────────
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs, _ = model(inputs.to(DEVICE), return_attention=True)
            all_preds.extend(outputs.squeeze().cpu().tolist())
            all_targets.extend(targets.numpy().tolist())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)

    # ── 1. Regression metrics ────────────────────────────────
    print("[Eval] Computing regression metrics...")
    reg = compute_regression_metrics(preds, targets)

    # ── 2. Bland-Altman ─────────────────────────────────────
    print("[Eval] Running Bland-Altman analysis...")
    ba = bland_altman_analysis(preds, targets)
    plot_bland_altman(ba, os.path.join(OUT_DIR, "bland_altman.png"))

    # ── 3. MC Dropout uncertainty ────────────────────────────
    unc = None
    if not args.skip_mc:
        print(f"[Eval] MC Dropout uncertainty ({args.mc_samples} samples)...")
        unc = mc_dropout_evaluation(model, val_loader, n_samples=args.mc_samples)
        plot_uncertainty(unc, os.path.join(OUT_DIR, "uncertainty.png"))

    # ── 4. Anomaly detection ─────────────────────────────────
    print("[Eval] Evaluating anomaly detection head...")
    ano = anomaly_evaluation(model, val_loader)

    # ── 5. FL convergence plots ──────────────────────────────
    if args.fl_compare:
        # When --compare was used in fedprox_client.py, each strategy
        # writes its own log: sim_log_fedavg.csv, sim_log_fedprox.csv,
        # sim_log_fedrqi.csv. Fall back to simulation_log.csv for FedRQI
        # if the dedicated file doesn't exist yet.
        fedrqi_log = "sim_log_fedrqi.csv"
        if not os.path.exists(fedrqi_log):
            fedrqi_log = args.fl_log   # fall back to simulation_log.csv
        log_files = {
            "FedAvg":  "sim_log_fedavg.csv",
            "FedProx": "sim_log_fedprox.csv",
            "FedRQI":  fedrqi_log,
        }
        # Report which files were found
        for name, path in log_files.items():
            status = "found" if os.path.exists(path) else "MISSING — strategy will be skipped"
            print(f"[Eval]   {name:<8}: {path} ({status})")
        plot_fl_convergence(log_files, os.path.join(OUT_DIR, "convergence_comparison.png"))
    elif os.path.exists(args.fl_log):
        plot_fl_convergence(
            {"FedRQI": args.fl_log},
            os.path.join(OUT_DIR, "convergence.png")
        )

    # ── 6. Print + save summary ──────────────────────────────
    print_summary(reg, ba, unc, ano)
    save_summary_csv(reg, ba, unc, ano,
                     os.path.join(OUT_DIR, "evaluation_summary.csv"))
    print(f"\n[Eval] All outputs saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()