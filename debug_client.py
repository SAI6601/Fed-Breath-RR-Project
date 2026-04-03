import torch
from client import BreathClient
import flwr as fl
from torch.utils.data import DataLoader, Subset, random_split
from dataset import BidmcDataset
from model import AttentionBiLSTM

# Initialize client
print("Initializing client...")
DEVICE = "cpu"
full_dataset = BidmcDataset()
train_data, val_data = random_split(full_dataset, [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))])
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

model = AttentionBiLSTM().to(DEVICE)
client = BreathClient(model, train_loader, val_loader, node_id=0)

weights = client.get_parameters({})

print("Calling fit...")
try:
    client.fit(parameters=weights, config={})
    print("Fit succeeded!")
except Exception as e:
    import traceback
    traceback.print_exc()
