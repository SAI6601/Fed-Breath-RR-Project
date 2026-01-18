import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
from dataset import BidmcDataset
from model import BreathBiLSTM

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_centralized():
    print(f"🚀 Starting Centralized Training on {device}...")
    
    # 1. Prepare Data
    full_dataset = BidmcDataset()
    
    # Split: 80% Training, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Data Loaded: {len(train_dataset)} Train samples, {len(val_dataset)} Validation samples.")

    # 2. Initialize Model, Loss, Optimizer
    model = BreathBiLSTM().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error (Standard for regression)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lists to store history for plotting
    train_losses = []
    val_maes = []

    # 3. Training Loop
    print("\nTraining in progress...")
    for epoch in range(EPOCHS):
        model.train() # Set to training mode
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape targets to match outputs [Batch, 1]
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass (Learn)
            loss.backward()
            
            # --- CRITICAL FIX: Gradient Clipping ---
            # This prevents the "Exploding Gradient" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
        # 4. Validation (Check accuracy)
        model.eval() # Set to evaluation mode
        val_mae_sum = 0.0
        with torch.no_grad(): # No need to track gradients for validation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Calculate Absolute Error (Difference in BrPM)
                mae = torch.abs(outputs - targets.unsqueeze(1))
                val_mae_sum += mae.sum().item()
        
        # Calculate averages
        avg_train_loss = running_loss / len(train_loader)
        avg_val_mae = val_mae_sum / len(val_dataset)
        
        train_losses.append(avg_train_loss)
        val_maes.append(avg_val_mae)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.2f} BrPM")

    # 5. Save the best model
    torch.save(model.state_dict(), "centralized_model.pth")
    print("\n✅ Training Complete! Model saved as 'centralized_model.pth'")
    
    # 6. Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss (MSE)')
    plt.plot(val_maes, label='Validation Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Progress: Loss & Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_centralized()