import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import flwr as fl
import argparse
import numpy as np
import os
from collections import OrderedDict
from scipy.signal import find_peaks

from dataset import BidmcDataset
from model import AttentionBiLSTM, rr_to_anomaly_label, ANOMALY_CLASSES

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
BATCH_SIZE       = 8
LEARNING_RATE    = 0.001
EPOCHS_PER_ROUND = 1
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED             = 42

# -- Differential Privacy -------------------------------------
DP_ENABLED       = True
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM    = 1.0
TARGET_DELTA     = 1e-5

# -- Multi-task loss weights -----------------------------------
LAMBDA_ANOMALY   = 0.3   # anomaly loss weight relative to RR regression loss

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def get_model_size_mb(model):
    tmp = f"temp_model_{os.getpid()}.p"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    if os.path.exists(tmp):
        os.remove(tmp)
    return size

def quantize_for_edge(model):
    model.to("cpu")
    return torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

def calculate_rqi_for_batch(ppg_batch):
    rqi_sum = 0.0
    signals = ppg_batch.cpu().numpy().squeeze(1)
    for sig in signals:
        peaks, _ = find_peaks(sig, distance=40, prominence=0.5)
        if len(peaks) < 2:
            rqi_sum += 0.1
            continue
        intervals = np.diff(peaks)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
        rqi_sum += max(0.0, 1.0 - cv)
    return rqi_sum / len(signals)

def get_params(model):
    base = getattr(model, '_module', model)
    return [val.cpu().numpy() for _, val in base.state_dict().items()]

def set_params(model, parameters):
    base = getattr(model, '_module', model)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(base.state_dict().keys(), parameters)}
    )
    base.load_state_dict(state_dict, strict=True)

# -------------------------------------------------------------
# Flower Client
# -------------------------------------------------------------
class BreathClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, node_id: int):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.node_id      = node_id
        self.criterion_rr = nn.MSELoss()
        self.criterion_ano = nn.CrossEntropyLoss()
        self.optimizer    = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.privacy_engine = None
        self.epsilon        = 0.0

        if DP_ENABLED:
            self._attach_privacy_engine()

    def _attach_privacy_engine(self):
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            if not ModuleValidator.is_valid(self.model):
                print(f"[DP] Node {self.node_id}: Fixing model for Opacus (LSTM -> DPLSTM)...")
                self.model = ModuleValidator.fix(self.model)
                self.model.to(DEVICE)
                self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = (
                self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=NOISE_MULTIPLIER,
                    max_grad_norm=MAX_GRAD_NORM,
                )
            )
            print(f"[DP] Node {self.node_id}: PrivacyEngine attached "
                  f"(sigma={NOISE_MULTIPLIER}, C={MAX_GRAD_NORM}, delta={TARGET_DELTA})")
        except ImportError:
            print("[DP] WARNING: Opacus not installed. Run: pip install opacus")
            self.privacy_engine = None
        except Exception as e:
            print(f"[DP] WARNING: PrivacyEngine setup failed ({e}). Continuing without DP.")
            self.privacy_engine = None

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        self.model.train()

        total_rqi    = 0.0
        batches      = 0
        anomaly_counts = [0] * len(ANOMALY_CLASSES)

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()

            # Multi-task forward pass (3 outputs)
            rr_pred, alpha, anomaly_logits = self.model(
                inputs, return_attention=True, return_anomaly=True
            )

            # -- Task 1: RR regression loss --
            rr_loss = self.criterion_rr(rr_pred, targets.unsqueeze(1))

            # -- Task 2: Anomaly classification loss --
            pseudo_labels = torch.tensor(
                [rr_to_anomaly_label(float(t)) for t in targets],
                dtype=torch.long, device=DEVICE
            )
            ano_loss = self.criterion_ano(anomaly_logits, pseudo_labels)

            # -- Combined loss --
            loss = rr_loss + LAMBDA_ANOMALY * ano_loss
            loss.backward()

            if self.privacy_engine is None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=MAX_GRAD_NORM
                )

            self.optimizer.step()

            preds = torch.argmax(anomaly_logits.detach(), dim=1).cpu().tolist()
            for p in preds:
                anomaly_counts[p] += 1

            total_rqi += calculate_rqi_for_batch(inputs)
            batches   += 1

        final_rqi = total_rqi / batches if batches > 0 else 0.0

        print(f"[Node] Node {self.node_id} | RQI: {final_rqi:.4f}")
        print(f"   Anomaly distribution this round:")
        for idx, cnt in enumerate(anomaly_counts):
            name = ANOMALY_CLASSES[idx]["name"]
            print(f"   {idx} {name:20s}: {cnt:4d} samples")

        if self.privacy_engine is not None:
            try:
                self.epsilon = self.privacy_engine.get_epsilon(delta=TARGET_DELTA)
                print(f"[DP] DP Budget: epsilon = {self.epsilon:.4f}, delta = {TARGET_DELTA}")
            except Exception as e:
                print(f"[DP] Could not compute epsilon: {e}")

        base_model      = getattr(self.model, '_module', self.model)
        quantized       = quantize_for_edge(base_model)
        fp32_size       = get_model_size_mb(base_model)
        int8_size       = get_model_size_mb(quantized)
        compression_pct = ((fp32_size - int8_size) / fp32_size) * 100
        print(f"[Edge] FP32={fp32_size:.3f}MB -> INT8={int8_size:.3f}MB ({compression_pct:.1f}%)")

        metrics = {
            "rqi":              float(final_rqi),
            "fp32_mb":          float(fp32_size),
            "int8_mb":          float(int8_size),
            "compression_pct":  float(compression_pct),
            "epsilon":          float(self.epsilon),
            "delta":            float(TARGET_DELTA),
            "dp_enabled":       float(1.0 if self.privacy_engine is not None else 0.0),
        }
        for idx, cnt in enumerate(anomaly_counts):
            metrics[f"anomaly_{idx}"] = float(cnt)

        return get_params(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        self.model.eval()
        loss, mae, steps = 0.0, 0.0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # return_anomaly=True returns (rr_pred, anomaly_logits)
                rr_pred, _ = self.model(inputs, return_anomaly=True)

                loss += self.criterion_rr(rr_pred, targets.unsqueeze(1)).item()
                mae  += torch.abs(rr_pred - targets.unsqueeze(1)).sum().item()
                steps += 1
                all_preds.extend(rr_pred.squeeze().cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        n          = len(self.val_loader.dataset)
        preds_np   = np.array(all_preds)
        targets_np = np.array(all_targets)
        rmse       = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))

        return float(loss / steps), n, {
            "mae":  float(mae / n),
            "rmse": rmse,
        }

    def personalize(self, local_loader, epochs=3, lr=1e-4):
        """
        Personalized FL (pFL): Fine-tune only the prediction heads
        while keeping the shared BiLSTM encoder frozen.

        This adapts the model to local patient demographics after
        global convergence, improving per-hospital accuracy without
        sharing head-layer gradients with the server.

        Args:
            local_loader: DataLoader with local hospital data
            epochs: number of fine-tuning epochs
            lr: learning rate for head layers
        """
        base_model = getattr(self.model, '_module', self.model)

        # Freeze shared encoder
        for p in base_model.lstm.parameters():
            p.requires_grad = False
        for p in base_model.attention_layer.parameters():
            p.requires_grad = False

        # Fine-tune heads only
        optimizer = optim.Adam([
            *base_model.fc_rr.parameters(),
            *base_model.fc_anomaly.parameters(),
        ], lr=lr)

        criterion_rr  = nn.MSELoss()
        criterion_ano = nn.CrossEntropyLoss()

        base_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in local_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()

                rr_pred, anomaly_logits = base_model(
                    inputs, return_anomaly=True
                )

                rr_loss = criterion_rr(rr_pred, targets.unsqueeze(1))
                pseudo = torch.tensor(
                    [rr_to_anomaly_label(float(t)) for t in targets],
                    dtype=torch.long, device=DEVICE
                )
                ano_loss = criterion_ano(anomaly_logits, pseudo)
                loss = rr_loss + LAMBDA_ANOMALY * ano_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            n_batches = max(len(local_loader), 1)
            print(f"[pFL] Node {self.node_id} | Epoch {epoch+1}/{epochs} | "
                  f"Loss: {total_loss/n_batches:.4f}")

        # Unfreeze for next global round
        for p in base_model.lstm.parameters():
            p.requires_grad = True
        for p in base_model.attention_layer.parameters():
            p.requires_grad = True

        print(f"[pFL] Personalization complete (Node {self.node_id})")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id",     type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Total number of clients for data partitioning (default: 2)")
    args = parser.parse_args()

    print(f"Starting Hospital Node #{args.node_id} on {DEVICE}")
    print(f"Differential Privacy : {'ENABLED' if DP_ENABLED else 'DISABLED'}")
    print(f"Anomaly Detection    : ENABLED (lambda={LAMBDA_ANOMALY})")

    full_dataset = BidmcDataset()
    num_clients  = args.num_clients
    split_size   = max(1, len(full_dataset) // num_clients)
    indices      = list(range(len(full_dataset)))
    start, end   = args.node_id * split_size, (args.node_id + 1) * split_size

    subset    = Subset(full_dataset, indices[start:end])
    train_len = int(0.8 * len(subset))
    val_len   = len(subset) - train_len
    generator = torch.Generator().manual_seed(SEED)
    train_data, val_data = random_split(subset, [train_len, val_len],
                                         generator=generator)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionBiLSTM().to(DEVICE)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8085",
        client=BreathClient(model, train_loader, val_loader, node_id=args.node_id),
    )

if __name__ == "__main__":
    main()