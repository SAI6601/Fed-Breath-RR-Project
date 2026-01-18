import torch
import torch.nn as nn

class BreathBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        """
        A Bidirectional LSTM Network for Respiratory Rate Estimation.
        
        Args:
            input_size (int): Number of features per time step (1 because we only use PPG).
            hidden_size (int): Number of neurons in the LSTM memory.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Final output (1 for Respiratory Rate).
        """
        super(BreathBiLSTM, self).__init__()
        
        # 1. The LSTM Layer (The core brain)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # Expect input shape: (Batch, Time, Features)
            bidirectional=True  # Read forwards and backwards
        )
        
        # 2. The Regressor (The final decision maker)
        # Since it's bidirectional, the hidden state size doubles (64 * 2 = 128)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Forward pass logic.
        x shape: (Batch_Size, 1, Sequence_Length) -> We need to transpose it.
        """
        # PyTorch LSTMs expect shape: (Batch, Sequence_Length, Features)
        # Our input is (Batch, Features, Sequence_Length) e.g., (1, 1, 3750)
        # So we swap dimensions 1 and 2.
        x = x.transpose(1, 2) 
        
        # Pass through LSTM
        # out shape: (Batch, Seq_Len, Hidden_Size*2)
        # _ (hidden_states): We ignore the internal memory states for now
        lstm_out, _ = self.lstm(x)
        
        # We only care about the LAST time step's output to make the prediction
        # Alternatively, we could average all time steps (Global Average Pooling)
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through the linear layer to get the final Respiratory Rate
        predicted_rr = self.fc(last_time_step)
        
        return predicted_rr

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("Testing Neural Network Architecture...")
    
    # 1. Create the model
    model = BreathBiLSTM()
    print("✅ Model created successfully.")
    print(model) # Prints the architecture
    
    # 2. Create a dummy input (1 batch, 1 channel, 3750 samples)
    # This simulates 30 seconds of PPG data
    dummy_input = torch.randn(1, 1, 3750)
    print(f"\nDummy Input Shape: {dummy_input.shape}")
    
    # 3. Pass it through the model
    output = model(dummy_input)
    
    # 4. Check output
    print(f"Output Shape: {output.shape} (Should be [1, 1])")
    print(f"Predicted RR: {output.item():.2f} (Random value, untrained)")
    
    if output.shape == (1, 1):
        print("\n✅ Success! Data flows through the network correctly.")
    else:
        print("\n❌ Error: Output shape mismatch.")