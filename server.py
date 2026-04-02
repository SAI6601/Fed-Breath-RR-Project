import flwr as fl
import csv
import os
import argparse
import torch
import numpy as np
from collections import OrderedDict
from model import AttentionBiLSTM

class FedRQI(fl.server.strategy.FedAvg):
    def __init__(self, *args, log_file="simulation_log.csv", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        # Initialize the CSV format with Anomaly Columns
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round","MAE","RMSE","C0_RQI","C1_RQI","FP32_MB","INT8_MB",
                             "Epsilon","Delta","DP_Enabled","Ano_0","Ano_1","Ano_2","Ano_3","Ano_4"])
        self.round_metrics = {}

    def aggregate_fit(self, server_round, results, failures):
        # --- BFT SECURITY SHIELD ---
        benign_results = []
        for client, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            l2_norm = sum(np.linalg.norm(w) for w in weights)

            # Threshold for malicious norm spike
            if l2_norm < 100.0:
                benign_results.append((client, fit_res))
            else:
                print(f"[BFT] SHIELD: Blocked anomalous update! L2 Norm: {l2_norm:.2f}")

        # Aggregate only benign weights
        aggregated_parameters, _ = super().aggregate_fit(server_round, benign_results, failures)

        if aggregated_parameters is not None:
            # --- 1. SAVE GLOBAL MODEL FOR APP.PY (REAL INFERENCE) ---
            try:
                weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
                model = AttentionBiLSTM()
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
                model.load_state_dict(state_dict, strict=True)
                torch.save(model.state_dict(), "centralized_model.pth")
            except Exception as e:
                print(f"[!!] Error saving global model: {e}")

            # --- 2. EXTRACT NEW METRICS ---
            c0_rqi, c1_rqi = 0.0, 0.0
            fp32_mb, int8_mb = 5.0, 1.2
            epsilon, delta, dp_enabled = 0.0, 1e-5, 0
            ano_counts = [0, 0, 0, 0, 0]

            for idx, (client, fit_res) in enumerate(benign_results):
                m = fit_res.metrics
                if idx == 0: c0_rqi = m.get("rqi", 0.0)
                if idx == 1: c1_rqi = m.get("rqi", 0.0)
                fp32_mb = m.get("fp32_mb", fp32_mb)
                int8_mb = m.get("int8_mb", int8_mb)
                epsilon = m.get("epsilon", epsilon)
                delta = m.get("delta", delta)
                dp_enabled = m.get("dp_enabled", dp_enabled)

                # Catch the anomaly distribution from clients
                for i in range(5):
                    ano_counts[i] += int(m.get(f"anomaly_{i}", 0))

            # Store metrics temporarily until evaluation phase
            self.round_metrics[server_round] = {
                "c0_rqi": c0_rqi, "c1_rqi": c1_rqi,
                "fp32": fp32_mb, "int8": int8_mb,
                "eps": epsilon, "delta": delta, "dp": dp_enabled,
                "anos": ano_counts
            }

        return aggregated_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return loss, metrics

        # Aggregate MAE & RMSE
        maes = [r.metrics["mae"] * r.num_examples for _, r in results]
        rmses = [r.metrics["rmse"] * r.num_examples for _, r in results]
        total_examples = sum(r.num_examples for _, r in results)

        agg_mae = sum(maes) / total_examples
        agg_rmse = sum(rmses) / total_examples

        # Retrieve fit metrics
        rm = self.round_metrics.get(server_round, {})
        anos = rm.get("anos", [0,0,0,0,0])

        # --- 3. WRITE EVERYTHING TO CSV ---
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                server_round, agg_mae, agg_rmse,
                rm.get("c0_rqi", 0.0), rm.get("c1_rqi", 0.0),
                rm.get("fp32", 5.0), rm.get("int8", 1.2),
                rm.get("eps", 0.0), rm.get("delta", 1e-5), rm.get("dp", 0),
                anos[0], anos[1], anos[2], anos[3], anos[4]
            ])

        return loss, {"mae": agg_mae}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fed-Breath FL Server")
    parser.add_argument("--strategy", default="fedrqi",
                        choices=["fedavg", "fedprox", "fedrqi"],
                        help="FL strategy to use (default: fedrqi)")
    parser.add_argument("--log-file", default="simulation_log.csv",
                        help="CSV log file path (default: simulation_log.csv)")
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="Number of FL rounds (default: 10)")
    args = parser.parse_args()

    print(f"[Server] Strategy: {args.strategy}")
    print(f"[Server] Rounds: {args.num_rounds}")
    print(f"[Server] Log: {args.log_file}")

    strategy = FedRQI(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        log_file=args.log_file,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8085",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy
    )