import flwr as fl
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar, FitRes

def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Aggregation function for evaluation metrics (like MAE).
    Calculates the weighted average based on number of samples.
    """
    total_samples = sum([num_examples for num_examples, _ in metrics])
    weighted_sum = sum([num_examples * m["mae"] for num_examples, m in metrics])
    return {"mae": weighted_sum / total_samples}

class FedRQI(fl.server.strategy.FedAvg):
    """
    Your Custom Strategy: RQI-Weighted Federated Averaging.
    Overrides the standard 'aggregate_fit' method.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # --- THE NOVELTY: Custom Weight Calculation ---
        # Standard FedAvg uses: weight = num_examples
        # FedRQI uses: weight = num_examples * RQI_Score
        
        total_weight = 0.0
        weighted_weights = []
        
        print(f"\n--- Round {server_round} Aggregation Report ---")
        
        # Iterate over results from all connected clients
        for _, fit_res in results:
            # 1. Unpack data
            num_examples = fit_res.num_examples
            # Get the RQI metric sent by the client (default to 1.0 if missing)
            client_rqi = fit_res.metrics.get("rqi", 1.0)
            
            # 2. Convert parameters (bytes) back to NumPy arrays
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            # 3. Calculate the "Quality-Aware Weight"
            # If RQI is low (0.2), this client's influence is slashed by 80%
            custom_weight = num_examples * client_rqi
            
            print(f"  > Client Update: {num_examples} samples | RQI: {client_rqi:.4f} | Influence: {custom_weight:.2f}")
            
            weighted_weights.append((parameters, custom_weight))
            total_weight += custom_weight

        # 4. Perform the weighted averaging of the model weights
        # Formula: Sum(Weight_i * Model_i) / Total_Weight
        aggregated_updates = [
            np.zeros_like(w) for w in weighted_weights[0][0]
        ]
        
        for weights, influence in weighted_weights:
            for i, layer_weight in enumerate(weights):
                aggregated_updates[i] += layer_weight * influence
        
        # Normalize by total weight
        aggregated_updates = [w / total_weight for w in aggregated_updates]
        
        # 5. Pack back into Flower Parameters format
        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_updates)
        
        return parameters_aggregated, {}

def main():
    parser = argparse.ArgumentParser(description="Fed-Breath Server")
    parser.add_argument("--strategy", type=str, default="fedrqi", choices=["fedavg", "fedrqi"], help="Choose aggregation strategy")
    args = parser.parse_args()

    # Define the Strategy
    if args.strategy == "fedrqi":
        print("🚀 Starting Server with NOVEL FedRQI Strategy (Quality-Weighted)...")
        strategy = FedRQI(
            fraction_fit=1.0,  # Sample 100% of available clients
            fraction_evaluate=1.0,
            min_fit_clients=2, # Wait for at least 2 clients to start
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average, # aggregate validation MAE
        )
    else:
        print("📉 Starting Server with Standard FedAvg (Baseline)...")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    # Start the Server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5), # Run for 5 rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    main()