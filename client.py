import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import flwr as fl
import argparse
import numpy as np
from collections import OrderedDict
from scipy.signal import find_peaks

# Import our custom modules
from dataset import BidmcDataset
from model import BreathBiLSTM

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS_PER_ROUND = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_rqi_for_batch(ppg_batch):
    """
    Calculates average RQI for a batch of signals.
    """
    rqi_sum = 0.0
    batch_size = ppg_batch.shape[0]
    
    signals = ppg_batch.cpu().numpy().squeeze(1)
    
    for i in range(batch_size):
        sig = signals[i]
        
        # FIX: Normalized data has mean 0 and std 1.
        # Peaks are usually > 0.5 (depending on wave).
        # We lower the prominence to catch smaller peaks.
        peaks, _ = find_peaks(sig, distance=40, prominence=0.5)
        
        if len(peaks) < 2:
            # If we can't find peaks, assign a low default score (0.1) instead of 0.0
            # This prevents the "Zero Influence" crash
            rqi_sum += 0.1
            continue
            
        intervals = np.diff(peaks)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            cv = 1.0
        else:
            cv = std_interval / mean_interval
            
        rqi = max(0.0, 1.0 - cv)
        rqi_sum += rqi
        
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
        # 1. Update Local Model
        set_params(self.model, parameters)
        
        # 2. Train Locally
        self.model.train()
        total_rqi = 0.0
        batches = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # --- CALCULATE REAL RQI ---
            batch_rqi = calculate_rqi_for_batch(inputs)
            total_rqi += batch_rqi
            batches += 1

        # Calculate average Quality Score for this hospital
        final_rqi = total_rqi / batches if batches > 0 else 0.0
        print(f"📊 Local Training Complete. Hospital RQI Score: {final_rqi:.4f}")

        # 3. Return updated weights + RQI
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
                outputs = self.model(inputs)
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
    num_clients = 3  # We will simulate 3 hospitals for the demo
    split_size = len(full_dataset) // num_clients
    
    indices = list(range(len(full_dataset)))
    start = args.node_id * split_size
    end = start + split_size
    
    subset = Subset(full_dataset, indices[start:end])
    train_len = int(0.8 * len(subset))
    train_data, val_data = random_split(subset, [train_len, len(subset)-train_len])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = BreathBiLSTM().to(DEVICE)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=BreathClient(model, train_loader, val_loader))

if __name__ == "__main__":
    main()