import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import flwr as fl
import argparse
import numpy as np
from collections import OrderedDict

# Import our custom modules
from dataset import BidmcDataset
from model import BreathBiLSTM

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS_PER_ROUND = 1  # In FL, we usually do 1-3 epochs per round
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_params(model):
    """Extracts model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model, parameters):
    """Updates the model with new weights from the server."""
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
        """Server asks: 'Give me your current weights'"""
        return get_params(self.model)

    def fit(self, parameters, config):
        """Server says: 'Train on your local data with these global weights'"""
        # 1. Update Local Model with Global Weights
        set_params(self.model, parameters)
        
        # 2. Train Locally
        self.model.train()
        running_loss = 0.0
        for _ in range(EPOCHS_PER_ROUND):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                
                # Clip Gradients (Safety from Day 5)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                running_loss += loss.item()

        # 3. Calculate RQI (The Novelty - Placeholder for now)
        # We will upgrade this on Day 8 to send the actual Quality Score
        avg_rqi = 1.0 

        # 4. Return updated weights to server
        # We send: Weights, Number of samples, and (Optionally) Metrics like RQI
        return get_params(self.model), len(self.train_loader.dataset), {"rqi": avg_rqi}

    def evaluate(self, parameters, config):
        """Server says: 'Test the global model on your local held-out data'"""
        set_params(self.model, parameters)
        self.model.eval()
        loss = 0.0
        mae = 0.0
        steps = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                
                # Calculate MSE Loss
                loss += self.criterion(outputs, targets.unsqueeze(1)).item()
                # Calculate MAE (Clinical metric)
                mae += torch.abs(outputs - targets.unsqueeze(1)).sum().item()
                steps += 1
                
        avg_loss = loss / steps
        avg_mae = mae / len(self.val_loader.dataset)
        
        return float(avg_loss), len(self.val_loader.dataset), {"mae": float(avg_mae)}

def main():
    # Parse Command Line Argument (Which "Hospital" is this?)
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--node-id", type=int, required=True, help="Client ID (0-9)")
    args = parser.parse_args()

    print(f"🏥 Starting Hospital Node #{args.node_id} on {DEVICE}...")

    # 1. Load Data
    full_dataset = BidmcDataset()
    
    # Simulate Federation: Split dataset into 10 chunks
    num_clients = 10
    total_size = len(full_dataset)
    split_size = total_size // num_clients
    
    # Calculate indices for this specific client
    # e.g., Client 0 gets indices 0-5, Client 1 gets 5-10...
    indices = list(range(len(full_dataset)))
    start_idx = args.node_id * split_size
    end_idx = (args.node_id + 1) * split_size if args.node_id < 9 else total_size
    
    my_indices = indices[start_idx:end_idx]
    my_subset = Subset(full_dataset, my_indices)
    
    # Split my subset into Train/Val (80/20)
    train_size = int(0.8 * len(my_subset))
    val_size = len(my_subset) - train_size
    train_data, val_data = random_split(my_subset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    print(f"📊 Local Data: {len(train_data)} Train, {len(val_data)} Val samples.")

    # 2. Initialize Model
    model = BreathBiLSTM().to(DEVICE)

    # 3. Start Client
    # This will sit and wait for the Server to give orders
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=BreathClient(model, train_loader, val_loader)
    )

if __name__ == "__main__":
    main()