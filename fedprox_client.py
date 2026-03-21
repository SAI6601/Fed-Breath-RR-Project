"""
fedprox_client.py — FedProx client variant for Fed-Breath

FedProx adds a proximal term to the local loss:
    L_total = L_task + (μ/2) * ||w - w_global||²

This penalises the local model for drifting too far from the
global model each round, which directly addresses the client
drift problem caused by non-IID respiratory data across hospitals.

Usage:
    # Start server (standard FedRQI server)
    python server.py

    # Start FedProx clients (replace client.py clients)
    python fedprox_client.py --node-id 0 --mu 0.01
    python fedprox_client.py --node-id 1 --mu 0.01

Strategy comparison (runs FedAvg → FedProx → FedRQI sequentially):
    python fedprox_client.py --compare
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import flwr as fl
import argparse
import numpy as np
import os
import csv
import subprocess
import sys
import time
from collections import OrderedDict
from scipy.signal import find_peaks

from dataset import BidmcDataset
from model import AttentionBiLSTM, rr_to_anomaly_label, ANOMALY_CLASSES

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BATCH_SIZE       = 8
LEARNING_RATE    = 0.001
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_ANOMALY   = 0.3
TARGET_DELTA     = 1e-5

# FedProx proximal term weight — key hyperparameter
# μ = 0 → reduces to FedAvg
# μ = 0.01 → mild regularisation (recommended for moderate non-IID)
# μ = 0.1  → strong regularisation (highly heterogeneous data)
DEFAULT_MU = 0.01

# ─────────────────────────────────────────────────────────────
# Helpers (identical to client.py)
# ─────────────────────────────────────────────────────────────
def get_params(model):
    base = getattr(model, '_module', model)
    return [val.cpu().numpy() for _, val in base.state_dict().items()]

def set_params(model, parameters):
    base = getattr(model, '_module', model)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(base.state_dict().keys(), parameters)}
    )
    base.load_state_dict(state_dict, strict=True)

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

# ─────────────────────────────────────────────────────────────
# FedProx Client
# ─────────────────────────────────────────────────────────────
class FedProxClient(fl.client.NumPyClient):
    """
    Extends standard FL training with the FedProx proximal term.

    The proximal term:
        (μ/2) * Σ ||w_k - w_global||²

    is computed as the squared L2 distance between the current
    local weights and the global weights received at the start
    of each round. This prevents aggressive local updates from
    pushing the model into a poor global optimum — especially
    important when different hospitals have very different
    patient populations (non-IID data).

    Reference: Li et al. "Federated Optimization in Heterogeneous
    Networks" (ICLR 2020). https://arxiv.org/abs/1812.06127
    """
    def __init__(self, model, train_loader, val_loader,
                 node_id: int, mu: float = DEFAULT_MU):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.node_id      = node_id
        self.mu           = mu
        self.criterion_rr  = nn.MSELoss()
        self.criterion_ano = nn.CrossEntropyLoss()
        self.optimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Holds a frozen copy of the global weights for proximal term
        self.global_params = None

    def get_parameters(self, config):
        return get_params(self.model)

    def _proximal_term(self) -> torch.Tensor:
        """
        Computes (μ/2) * ||w_local - w_global||²
        Returns scalar tensor on the model's device.
        """
        if self.global_params is None or self.mu == 0.0:
            return torch.tensor(0.0, device=DEVICE)

        prox = torch.tensor(0.0, device=DEVICE)
        local_params = list(self.model.parameters())

        for local_w, global_w in zip(local_params, self.global_params):
            prox += torch.sum((local_w - global_w.to(DEVICE)) ** 2)

        return (self.mu / 2.0) * prox

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Freeze a copy of global weights for the proximal term
        # detach() ensures no gradients flow through the reference copy
        self.global_params = [
            torch.tensor(p).detach().to(DEVICE)
            for p in parameters
        ]

        self.model.train()
        total_rqi      = 0.0
        batches        = 0
        anomaly_counts = [0] * len(ANOMALY_CLASSES)

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()

            # ── Multi-task forward pass ──────────────────────────
            rr_pred, alpha, anomaly_logits = self.model(
                inputs, return_attention=True, return_anomaly=True
            )

            # ── Task losses ──────────────────────────────────────
            rr_loss  = self.criterion_rr(rr_pred, targets.unsqueeze(1))
            pseudo   = torch.tensor(
                [rr_to_anomaly_label(float(t)) for t in targets],
                dtype=torch.long, device=DEVICE
            )
            ano_loss = self.criterion_ano(anomaly_logits, pseudo)

            # ── FedProx proximal term ────────────────────────────
            prox_loss = self._proximal_term()

            # ── Combined loss ────────────────────────────────────
            loss = rr_loss + LAMBDA_ANOMALY * ano_loss + prox_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            preds = torch.argmax(anomaly_logits.detach(), dim=1).cpu().tolist()
            for p in preds:
                anomaly_counts[p] += 1

            total_rqi += calculate_rqi_for_batch(inputs)
            batches   += 1

        final_rqi = total_rqi / batches if batches > 0 else 0.0
        prox_val  = self._proximal_term().item()

        print(f"[FedProx] Node {self.node_id} | RQI: {final_rqi:.4f} | "
              f"ProxTerm: {prox_val:.6f} | μ={self.mu}")

        metrics = {
            "rqi":        float(final_rqi),
            "prox_term":  float(prox_val),
            "mu":         float(self.mu),
            "strategy":   0.0,   # 1=FedProx marker for server logging
            "epsilon":    0.0,
            "delta":      float(TARGET_DELTA),
            "dp_enabled": 0.0,
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
                rr_pred, _ = self.model(inputs, return_anomaly=True)
                loss  += self.criterion_rr(rr_pred, targets.unsqueeze(1)).item()
                mae   += torch.abs(rr_pred - targets.unsqueeze(1)).sum().item()
                steps += 1
                all_preds.extend(rr_pred.squeeze().cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        n    = len(self.val_loader.dataset)
        preds_np   = np.array(all_preds)
        targets_np = np.array(all_targets)
        rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))

        return float(loss / steps), n, {
            "mae":  float(mae / n),
            "rmse": rmse,
        }


# ─────────────────────────────────────────────────────────────
# Strategy comparison runner
# ─────────────────────────────────────────────────────────────
COMPARISON_LOG = "strategy_comparison.csv"

def _wait_for_log(log_file: str, timeout: int = 300) -> bool:
    """
    Polls until log_file has at least one data row, or timeout seconds pass.
    Returns True if data appeared, False if timed out.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_file):
            with open(log_file) as f:
                rows = list(csv.reader(f))
            if len(rows) > 1:   # header + at least 1 data row
                return True
        time.sleep(2)
    return False


def _parse_log(log_file: str) -> tuple:
    """Parse MAE and RMSE columns from a simulation log CSV."""
    maes, rmses = [], []
    if not os.path.exists(log_file):
        return maes, rmses
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                maes.append(float(row["MAE"]))
                # RMSE column added in updated server.py — fall back to MAE if absent
                rmses.append(float(row.get("RMSE") or row["MAE"]))
            except (KeyError, ValueError):
                pass
    return maes, rmses


def run_strategy_comparison():
    """
    Runs three FL experiments back-to-back and saves a comparison
    CSV with per-round MAE and RMSE for each strategy:
      • FedAvg  (μ = 0, no RQI weighting)
      • FedProx (μ = 0.01, no RQI weighting)
      • FedRQI  (μ = 0, RQI-weighted + BFT — our full system)

    Results are saved to strategy_comparison.csv and printed as a
    summary table — ready to paste into a paper as Table II.

    Subprocess output is written to per-strategy log files
    (server_fedavg.log, etc.) so errors are always visible.
    """
    print("\n" + "="*60)
    print("  STRATEGY COMPARISON: FedAvg vs FedProx vs FedRQI")
    print("="*60)

    # Verify data directory exists before launching anything
    if not os.path.isdir(os.path.join("data", "raw")):
        print("\n[ERROR] data/raw/ not found.")
        print("        Download BIDMC dataset and place it in data/raw/")
        print("        before running the strategy comparison.")
        return

    strategies = [
        {"name": "FedAvg",  "strategy_flag": "fedavg",  "mu": 0.0},
        {"name": "FedProx", "strategy_flag": "fedprox", "mu": 0.01},
        {"name": "FedRQI",  "strategy_flag": "fedrqi",  "mu": 0.0},
    ]

    results = {}
    TIMEOUT = 600   # seconds per strategy (10 min is generous for 5 rounds)

    # Force UTF-8 I/O in subprocesses — prevents UnicodeEncodeError on Windows
    # when print() outputs emoji characters into redirected log files.
    utf8_env = os.environ.copy()
    utf8_env["PYTHONIOENCODING"] = "utf-8"
    utf8_env["PYTHONUTF8"]       = "1"   # Python 3.7+ UTF-8 mode flag

    for s in strategies:
        name     = s["name"]
        log_file = f"sim_log_{s['strategy_flag']}.csv"
        srv_log  = open(f"server_{s['strategy_flag']}.log", "w", encoding="utf-8")
        cli_logs = []

        print(f"\n▶ Running {name} (μ={s['mu']})...")
        print(f"  Server log  → server_{s['strategy_flag']}.log")
        print(f"  Results log → {log_file}")

        if os.path.exists(log_file):
            os.remove(log_file)

        # ── Launch server ────────────────────────────────────
        server_proc = subprocess.Popen(
            [sys.executable, "server.py",
             "--strategy",   s["strategy_flag"],
             "--log-file",   log_file,
             "--num-rounds", "20"],
            stdout=srv_log, stderr=srv_log,
            env=utf8_env
        )

        # Give server enough time to bind the port
        print("  Waiting for server to start (5 s)...", end="", flush=True)
        time.sleep(5)

        if server_proc.poll() is not None:
            print(f" FAILED (exit {server_proc.returncode})")
            print(f"  Check server_{s['strategy_flag']}.log for details.")
            srv_log.close()
            results[name] = {"maes": [], "rmses": [], "final_mae": float("nan"), "final_rmse": float("nan")}
            continue
        print(" OK")

        # ── Launch clients ───────────────────────────────────
        client_procs = []
        for node_id in range(2):
            cli_log = open(f"client_{s['strategy_flag']}_{node_id}.log", "w", encoding="utf-8")
            cli_logs.append(cli_log)
            script = "fedprox_client.py" if s["mu"] > 0 else "client.py"
            cmd = [sys.executable, script, "--node-id", str(node_id)]
            if s["mu"] > 0:
                cmd += ["--mu", str(s["mu"])]
            client_procs.append(
                subprocess.Popen(cmd, stdout=cli_log, stderr=cli_log, env=utf8_env)
            )
            print(f"  Client {node_id} started → client_{s['strategy_flag']}_{node_id}.log")

        # ── Wait for results to appear ───────────────────────
        print(f"  Running FL ({TIMEOUT}s timeout)...", end="", flush=True)
        data_arrived = _wait_for_log(log_file, timeout=TIMEOUT)

        if not data_arrived:
            print(" TIMED OUT")
            print(f"  Check server_{s['strategy_flag']}.log and client logs for errors.")
        else:
            print(" done")

        # ── Wait for server to finish cleanly ────────────────
        try:
            server_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server_proc.kill()

        for p in client_procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

        # Close log file handles
        srv_log.close()
        for f in cli_logs:
            f.close()

        # ── Parse results ────────────────────────────────────
        maes, rmses = _parse_log(log_file)
        results[name] = {
            "maes":       maes,
            "rmses":      rmses,
            "final_mae":  maes[-1]  if maes  else float("nan"),
            "final_rmse": rmses[-1] if rmses else float("nan"),
        }

        mae_str  = f"{results[name]['final_mae']:.4f}"  if maes  else "nan (no data)"
        rmse_str = f"{results[name]['final_rmse']:.4f}" if rmses else "nan (no data)"
        print(f"  {name} — Final MAE: {mae_str} | RMSE: {rmse_str}")

    # ── Write comparison CSV ─────────────────────────────────
    with open(COMPARISON_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Round", "MAE", "RMSE"])
        for name, data in results.items():
            for i, (mae, rmse) in enumerate(zip(data["maes"], data["rmses"]), 1):
                writer.writerow([name, i, round(mae, 4), round(rmse, 4)])

    # ── Print summary table ──────────────────────────────────
    print("\n" + "="*60)
    print(f"  {'Strategy':<12} {'Final MAE':>12} {'Final RMSE':>12}")
    print("-"*40)
    for name, data in results.items():
        mae_s  = f"{data['final_mae']:.4f}"  if not (data['final_mae']  != data['final_mae']) else "  nan"
        rmse_s = f"{data['final_rmse']:.4f}" if not (data['final_rmse'] != data['final_rmse']) else "  nan"
        print(f"  {name:<12} {mae_s:>12} {rmse_s:>12}")
    print("="*60)
    print(f"\nFull results saved to : {COMPARISON_LOG}")
    print("Per-run logs saved as  : server_<strategy>.log, client_<strategy>_<id>.log")
    print("(Ready to use as Table II in your paper)")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FedProx client for Fed-Breath")
    parser.add_argument("--node-id", type=int, default=0,
                        help="Client node ID (0 or 1)")
    parser.add_argument("--mu", type=float, default=DEFAULT_MU,
                        help=f"FedProx proximal term weight (default: {DEFAULT_MU})")
    parser.add_argument("--compare", action="store_true",
                        help="Run full strategy comparison experiment")
    args = parser.parse_args()

    if args.compare:
        run_strategy_comparison()
        return

    print(f"🏥 Starting FedProx Node #{args.node_id} on {DEVICE}")
    print(f"📐 Proximal term μ = {args.mu}  ({'FedProx' if args.mu > 0 else 'reduces to FedAvg'})")

    full_dataset = BidmcDataset()
    num_clients  = 2
    split_size   = max(1, len(full_dataset) // num_clients)
    indices      = list(range(len(full_dataset)))
    start, end   = args.node_id * split_size, (args.node_id + 1) * split_size

    subset    = Subset(full_dataset, indices[start:end])
    train_len = int(0.8 * len(subset))
    val_len   = len(subset) - train_len
    train_data, val_data = random_split(subset, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionBiLSTM().to(DEVICE)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8085",
        client=FedProxClient(model, train_loader, val_loader,
                             node_id=args.node_id, mu=args.mu),
    )

if __name__ == "__main__":
    main()