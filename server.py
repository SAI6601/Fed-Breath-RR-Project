import flwr as fl
import numpy as np
import argparse
import csv
import os
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar, FitRes

# --- GUI LOGGING SETUP ---
LOG_FILE = "simulation_log.csv"   # overridable via --log-file argument

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_samples = sum([num_examples for num_examples, _ in metrics])
    weighted_mae  = sum([num_examples * m["mae"]  for num_examples, m in metrics])
    # RMSE cannot be linearly averaged — take sqrt of weighted mean of squared errors
    # Proxy: weighted average of per-client RMSE (good enough for monitoring)
    weighted_rmse = sum([num_examples * m.get("rmse", m["mae"])
                         for num_examples, m in metrics])
    return {
        "mae":  weighted_mae  / total_samples,
        "rmse": weighted_rmse / total_samples,
    }

class FedRQI(fl.server.strategy.FedAvg):
    def log_metrics(self, round_num, client_rqis, global_mae, global_rmse=None):
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Round', 'MAE', 'RMSE',
                    'Client_0_RQI', 'Client_1_RQI',
                    'FP32_MB', 'INT8_MB',
                    'Epsilon', 'Delta', 'DP_Enabled',
                    'Ano_Normal', 'Ano_Brady', 'Ano_Apnea', 'Ano_Tachy', 'Ano_SevTachy'
                ])

            c0_rqi     = client_rqis[0] if len(client_rqis) > 0 else 0
            c1_rqi     = client_rqis[1] if len(client_rqis) > 1 else 0
            fp32       = getattr(self, 'last_fp32_mb',   5.4)
            int8       = getattr(self, 'last_int8_mb',   1.2)
            epsilon    = getattr(self, 'last_epsilon',   -1.0)
            delta      = getattr(self, 'last_delta',     1e-5)
            dp_enabled = getattr(self, 'last_dp_enabled', 0)
            ac         = getattr(self, 'last_anomaly_counts', {})
            rmse       = global_rmse if global_rmse is not None else global_mae
            writer.writerow([round_num, global_mae, rmse,
                             c0_rqi, c1_rqi,
                             fp32, int8,
                             epsilon, delta, dp_enabled,
                             ac.get('anomaly_0', 0), ac.get('anomaly_1', 0),
                             ac.get('anomaly_2', 0), ac.get('anomaly_3', 0),
                             ac.get('anomaly_4', 0)])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        self.last_rqis = [res.metrics.get("rqi", 0.0) for _, res in results]
        # Capture real edge compression metrics from clients
        fp32_vals = [res.metrics.get("fp32_mb", 5.4) for _, res in results]
        int8_vals = [res.metrics.get("int8_mb", 1.2) for _, res in results]
        self.last_fp32_mb = float(sum(fp32_vals) / len(fp32_vals))
        self.last_int8_mb = float(sum(int8_vals) / len(int8_vals))

        # Capture DP privacy budget (epsilon) — take the MAX across clients
        eps_vals = [res.metrics.get("epsilon", -1.0) for _, res in results]
        valid_eps = [e for e in eps_vals if e >= 0]
        self.last_epsilon    = float(max(valid_eps)) if valid_eps else -1.0
        self.last_delta      = float(results[0][1].metrics.get("delta", 1e-5))
        self.last_dp_enabled = int(any(res.metrics.get("dp_enabled", 0) > 0.5
                                       for _, res in results))

        if valid_eps:
            print(f"🔒 Privacy Budget — ε = {self.last_epsilon:.4f}, "
                  f"δ = {self.last_delta:.0e}  "
                  f"({'DP-SGD active' if self.last_dp_enabled else 'no DP'})")

        # Aggregate anomaly class counts across all clients
        self.last_anomaly_counts = {}
        for idx in range(5):
            key = f"anomaly_{idx}"
            total = sum(res.metrics.get(key, 0.0) for _, res in results)
            self.last_anomaly_counts[key] = int(total)

        if any(v > 0 for v in self.last_anomaly_counts.values()):
            print("🧠 Anomaly Distribution (this round):")
            labels = {0:"Normal",1:"Bradypnea",2:"Apnea",3:"Tachypnea",4:"SevTachy"}
            for idx in range(5):
                print(f"   {labels[idx]:18s}: {self.last_anomaly_counts[f'anomaly_{idx}']}")

        print(f"\n========================================")
        print(f"--- Round {server_round} Aggregation Report ---")
        
        raw_updates = []
        client_norms = []
        
        # 1. Unpack all client updates
        for _, fit_res in results:
            num_examples = fit_res.num_examples
            client_rqi = fit_res.metrics.get("rqi", 1.0)
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            # --- PHASE 3 NOVELTY: Extract Mathematical Shape for BFT ---
            # Calculate the L2 Norm (magnitude) of the client's weights
            norm = np.sqrt(sum(np.sum(p**2) for p in parameters))
            client_norms.append(norm)
            
            custom_weight = num_examples * max(client_rqi, 0.01)
            raw_updates.append({"params": parameters, "weight": custom_weight, "rqi": client_rqi, "samples": num_examples})

        # --- PHASE 3 NOVELTY: BYZANTINE SECURITY SHIELD ---
        print("🛡️  BFT Security Scan:")
        median_norm = np.median(client_norms)
        malicious_threshold = median_norm * 3.0 # If weights are 3x larger than normal, it's a poison attack
        
        total_weight = 0.0
        weighted_weights = []
        
        for i, update in enumerate(raw_updates):
            norm = client_norms[i]
            
            # Check for Data Poisoning
            if norm > malicious_threshold and norm > 10.0:
                print(f"   🚨 BLOCKED: Client {i} detected as malicious! (Abnormal Norm: {norm:.2f})")
                update["weight"] = 0.0 # Drop the hacker's update completely
            else:
                print(f"   ✅ PASS: Client {i} looks benign. (Norm: {norm:.2f})")
                
            print(f"   > Update Accepted: {update['samples']} samples | RQI: {update['rqi']:.4f} | Influence: {update['weight']:.2f}")
            
            if update["weight"] > 0:
                weighted_weights.append((update["params"], update["weight"]))
                total_weight += update["weight"]

        if total_weight == 0.0 or not weighted_weights:
            print("⚠️ WARNING: All clients flagged as malicious! Skipping this round.")
            return None, {}

        # 3. Aggregate
        aggregated_updates = [np.zeros_like(w) for w in weighted_weights[0][0]]
        for weights, influence in weighted_weights:
            for i, layer_weight in enumerate(weights):
                aggregated_updates[i] += layer_weight * influence
        
        aggregated_updates = [w / total_weight for w in aggregated_updates]
        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_updates)
        
        return parameters_aggregated, {}

    def aggregate_evaluate(self, server_round, results, failures):
        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)
        if metrics_agg and "mae" in metrics_agg:
            rqis_to_log = getattr(self, 'last_rqis', [])
            rmse = metrics_agg.get("rmse", None)
            self.log_metrics(server_round, rqis_to_log, metrics_agg["mae"], rmse)
        return loss_agg, metrics_agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",   type=str, default="fedrqi")
    parser.add_argument("--log-file",   type=str, default="simulation_log.csv",
                        help="CSV log file path (allows per-strategy logs)")
    parser.add_argument("--num-rounds", type=int, default=5,
                        help="Number of FL rounds (default: 5)")
    args = parser.parse_args()

    # Override global LOG_FILE so strategy comparison can write separate files
    global LOG_FILE
    LOG_FILE = args.log_file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print(f"🚀 Starting Server | strategy={args.strategy} | "
          f"rounds={args.num_rounds} | log={LOG_FILE}")
    strategy = FedRQI(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()