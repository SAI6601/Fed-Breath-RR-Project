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

# Import our custom modules
from dataset import BidmcDataset
# --- PHASE 1 UPDATE: Import the new Explainable Model ---
from model import AttentionBiLSTM

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS_PER_ROUND = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_size_mb(model):
    """Helper function to measure model size in Megabytes."""
    torch.save(model.state_dict(), "temp_model.p")
    size = os.path.getsize("temp_model.p") / 1e6
    if os.path.exists("temp_model.p"):
        os.remove("temp_model.p")
    return size

def quantize_for_edge(model):
    """
    Applies Post-Training Dynamic Quantization (PTQ).
    Converts FP32 weights to INT8 integers for Edge IoT devices.
    """
    # Quantization typically targets CPU/Edge deployment
    model.to("cpu") 
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def calculate_rqi_for_batch(ppg_batch):
    """Calculates average RQI for a batch of signals."""
    rqi_sum = 0.0
    batch_size = ppg_batch.shape[0]
    signals = ppg_batch.cpu().numpy().squeeze(1)
    
    for i in range(batch_size):
        sig = signals[i]
        peaks, _ = find_peaks(sig, distance=40, prominence=0.5) 
        
        if len(peaks) < 2:
            rqi_sum += 0.1 
            continue
            
        intervals = np.diff(peaks)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        cv = 1.0 if mean_interval == 0 else std_interval / mean_interval
        rqi_sum += max(0.0, 1.0 - cv)
        
    return rqi_sum / batch_size

def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class BreathClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        self.model.train()
        total_rqi = 0.0
        batches = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            
            # --- PHASE 1 UPDATE: Handle the XAI Attention outputs ---
            outputs, _ = self.model(inputs, return_attention=True)
            
            loss = self.criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_rqi += calculate_rqi_for_batch(inputs)
            batches += 1

        final_rqi = total_rqi / batches if batches > 0 else 0.0
        print(f"📊 Local Training Complete. Hospital RQI Score: {final_rqi:.4f}")

        # --- PHASE 2 NOVELTY: EDGE OPTIMIZATION ---
        quantized_edge_model = quantize_for_edge(self.model)
        fp32_size = get_model_size_mb(self.model)
        int8_size = get_model_size_mb(quantized_edge_model)
        
        print(f"⚙️  Edge Optimization (Quantization):")
        print(f"   > FP32 Model Size: {fp32_size:.4f} MB")
        print(f"   > INT8 Model Size: {int8_size:.4f} MB")
        print(f"   > Memory Reduced by: {((fp32_size - int8_size) / fp32_size) * 100:.1f}%")

        return get_params(self.model), len(self.train_loader.dataset), {"rqi": float(final_rqi)}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        self.model.eval()
        loss = 0.0
        mae = 0.0
        steps = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # --- PHASE 1 UPDATE ---
                outputs, _ = self.model(inputs, return_attention=True)
                
                loss += self.criterion(outputs, targets.unsqueeze(1)).item()
                mae += torch.abs(outputs - targets.unsqueeze(1)).sum().item()
                steps += 1
        return float(loss/steps), len(self.val_loader.dataset), {"mae": float(mae/len(self.val_loader.dataset))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=int, required=True)
    args = parser.parse_args()

    print(f"🏥 Starting Hospital Node #{args.node_id} on {DEVICE}...")

    # Load and partition data
    full_dataset = BidmcDataset()
    num_clients = 2
    split_size = max(1, len(full_dataset) // num_clients)
    
    indices = list(range(len(full_dataset)))
    start = args.node_id * split_size
    end = start + split_size
    
    subset = Subset(full_dataset, indices[start:end])
    train_len = int(0.8 * len(subset))
    val_len = len(subset) - train_len
    
    train_data, val_data = random_split(subset, [train_len, val_len])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Instantiate the new XAI model
    model = AttentionBiLSTM().to(DEVICE)
    
    fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=BreathClient(model, train_loader, val_loader))

if __name__ == "__main__":
    main()