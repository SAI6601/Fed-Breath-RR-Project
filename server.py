import flwr as fl
import numpy as np
import argparse
import csv
import os
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar, FitRes

# --- GUI LOGGING SETUP ---
LOG_FILE = "gui_data.csv"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE) # Reset log on start
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "MAE", "Client_0_RQI", "Client_1_RQI"])

def write_to_log(round_num, mae, rqis):
    # Ensure we have at least 2 clients worth of data
    c0_rqi = rqis[0] if len(rqis) > 0 else 0
    c1_rqi = rqis[1] if len(rqis) > 1 else 0
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([round_num, mae, c0_rqi, c1_rqi])

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total_samples = sum([num_examples for num_examples, _ in metrics])
    weighted_sum = sum([num_examples * m["mae"] for num_examples, m in metrics])
    return {"mae": weighted_sum / total_samples}

class FedRQI(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}
        
        # 1. Capture RQIs for the GUI
        client_rqis = [res.metrics.get("rqi", 0.0) for _, res in results]
        self.last_rqis = client_rqis # Store them for the evaluate step
        
        # ... [Keep your existing RQI logic here] ...
        # (Copy the Divide-by-Zero fix logic from Day 8 here)
        total_weight = 0.0
        weighted_weights = []
        
        for _, fit_res in results:
            num_examples = fit_res.num_examples
            client_rqi = fit_res.metrics.get("rqi", 1.0)
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            custom_weight = num_examples * max(client_rqi, 0.01)
            weighted_weights.append((parameters, custom_weight))
            total_weight += custom_weight

        if total_weight == 0.0:
            total_weight = 1.0
            weighted_weights = [(w, 1.0) for w, _ in weighted_weights]
            total_weight = len(weighted_weights)

        aggregated_updates = [np.zeros_like(w) for w in weighted_weights[0][0]]
        for weights, influence in weighted_weights:
            for i, layer_weight in enumerate(weights):
                aggregated_updates[i] += layer_weight * influence
        
        aggregated_updates = [w / total_weight for w in aggregated_updates]
        return fl.common.ndarrays_to_parameters(aggregated_updates), {}

    def aggregate_evaluate(self, server_round, results, failures):
        # 1. Run standard aggregation to get MAE
        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)
        
        # 2. Write to GUI Log
        if metrics_agg and "mae" in metrics_agg:
            write_to_log(server_round, metrics_agg["mae"], getattr(self, 'last_rqis', []))
            
        return loss_agg, metrics_agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="fedrqi")
    args = parser.parse_args()

    strategy = FedRQI(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()