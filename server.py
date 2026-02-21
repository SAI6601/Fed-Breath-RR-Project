import flwr as fl
import numpy as np
import argparse
import csv
import os
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar, FitRes

# --- GUI LOGGING SETUP ---
LOG_FILE = "simulation_log.csv"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_samples = sum([num_examples for num_examples, _ in metrics])
    weighted_sum = sum([num_examples * m["mae"] for num_examples, m in metrics])
    return {"mae": weighted_sum / total_samples}

class FedRQI(fl.server.strategy.FedAvg):
    def log_metrics(self, round_num, client_rqis, global_mae):
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Round', 'MAE', 'Client_0_RQI', 'Client_1_RQI'])
            
            c0_rqi = client_rqis[0] if len(client_rqis) > 0 else 0
            c1_rqi = client_rqis[1] if len(client_rqis) > 1 else 0
            writer.writerow([round_num, global_mae, c0_rqi, c1_rqi])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        self.last_rqis = [res.metrics.get("rqi", 0.0) for _, res in results]

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

        if total_weight == 0.0:
            print("⚠️ WARNING: Total Weight is 0! Falling back to uniform averaging.")
            total_weight = 1.0
            weighted_weights = [(w, 1.0) for w, _ in weighted_weights]
            total_weight = len(weighted_weights)

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
            self.log_metrics(server_round, rqis_to_log, metrics_agg["mae"])
        return loss_agg, metrics_agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="fedrqi")
    args = parser.parse_args()

    print("🚀 Starting Server with NOVEL FedRQI Strategy + BFT Shield...")
    strategy = FedRQI(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8085", # Using our new safe port
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()