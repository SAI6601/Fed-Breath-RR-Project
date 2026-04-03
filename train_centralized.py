import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — safe on Windows/servers
import matplotlib.pyplot as plt

from dataset import BidmcDataset
from model import AttentionBiLSTM, rr_to_anomaly_label

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BATCH_SIZE    = 8
LEARNING_RATE = 0.001
EPOCHS        = 20
LAMBDA_ANOMALY = 0.3   # anomaly loss weight (matches client.py)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_centralized():
    print(f"Starting centralized training on {DEVICE}...")
    print(f"Multi-task loss: MSE(RR) + {LAMBDA_ANOMALY} x CE(anomaly)")

    # ── Data ────────────────────────────────────────────────────
    full_dataset = BidmcDataset()
    train_size   = int(0.8 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # drop_last=True keeps batch size consistent (needed if Opacus is used later)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data: {len(train_ds)} train / {len(val_ds)} val samples")

    # ── Model + losses ───────────────────────────────────────────
    model         = AttentionBiLSTM().to(DEVICE)
    criterion_rr  = nn.MSELoss()
    criterion_ano = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training history ─────────────────────────────────────────
    history = {
        "train_loss": [], "train_rr_loss": [], "train_ano_loss": [],
        "val_mae": [],    "val_rmse": [],
    }
    best_mae  = float("inf")
    best_path = "centralized_model.pth"

    for epoch in range(EPOCHS):
        # ── Train ───────────────────────────────────────────────
        model.train()
        total_loss = rr_loss_sum = ano_loss_sum = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # Multi-task forward pass
            rr_pred, _, anomaly_logits = model(
                inputs, return_attention=True, return_anomaly=True
            )

            # RR regression loss
            rr_loss = criterion_rr(rr_pred, targets.unsqueeze(1))

            # Anomaly classification loss (pseudo-labels from RR targets)
            pseudo = torch.tensor(
                [rr_to_anomaly_label(float(t)) for t in targets],
                dtype=torch.long, device=DEVICE
            )
            ano_loss = criterion_ano(anomaly_logits, pseudo)

            loss = rr_loss + LAMBDA_ANOMALY * ano_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss   += loss.item()
            rr_loss_sum  += rr_loss.item()
            ano_loss_sum += ano_loss.item()

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                rr_pred, _ = model(inputs, return_attention=True)
                all_preds.extend(rr_pred.squeeze().cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        preds_np   = np.array(all_preds)
        tgts_np    = np.array(all_targets)
        val_mae    = float(np.mean(np.abs(preds_np - tgts_np)))
        val_rmse   = float(np.sqrt(np.mean((preds_np - tgts_np) ** 2)))

        n = max(len(train_loader), 1)
        history["train_loss"].append(total_loss / n)
        history["train_rr_loss"].append(rr_loss_sum / n)
        history["train_ano_loss"].append(ano_loss_sum / n)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {total_loss/n:.4f}  "
              f"(RR: {rr_loss_sum/n:.4f}  Ano: {ano_loss_sum/n:.4f}) | "
              f"Val MAE: {val_mae:.3f}  RMSE: {val_rmse:.3f} BrPM")

        # ── Save best checkpoint ──────────────────────────────────
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_path)
            print(f"           -> New best MAE {best_mae:.3f} — saved to {best_path}")

    print(f"\nTraining complete. Best Val MAE: {best_mae:.4f} BrPM")
    print(f"Model saved to: {best_path}")

    # ── Learning curve plot ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_rr_loss"],  label="RR loss (MSE)",      color="#378ADD", linewidth=1.8)
    axes[0].plot(history["train_ano_loss"], label="Anomaly loss (CE)",   color="#EF9F27", linewidth=1.8)
    axes[0].plot(history["train_loss"],     label="Total loss",          color="#E24B4A", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training losses (multi-task)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_mae"],  label="Val MAE",  color="#1D9E75", linewidth=1.8)
    axes[1].plot(history["val_rmse"], label="Val RMSE", color="#534AB7", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (BrPM)")
    axes[1].set_title("Validation error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = "training_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Learning curve saved to: {out}")

if __name__ == "__main__":
    train_centralized()