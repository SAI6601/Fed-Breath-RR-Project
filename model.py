import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        """
        Upgraded Bidirectional LSTM with an Attention Mechanism for Explainable AI (XAI).
        """
        super(AttentionBiLSTM, self).__init__()
        
        # 1. The Core Memory
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. THE NOVELTY: The Attention Layer
        # BiLSTM outputs double the hidden size (forward + backward)
        self.attention_layer = nn.Linear(hidden_size * 2, 1)
        
        # 3. The Final Regressor
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, return_attention=False):
        """
        x shape: (Batch_Size, Features, Sequence_Length)
        """
        # Swap dimensions to match PyTorch LSTM expectations -> (Batch, Seq_Len, Features)
        x = x.transpose(1, 2) 
        
        # Pass through LSTM
        # lstm_out shape: (Batch, Seq_Len, Hidden_Size*2)
        lstm_out, _ = self.lstm(x)
        
        # --- ATTENTION MECHANISM MATH ---
        # 1. Score each time step: How important is this specific millisecond?
        attention_scores = torch.tanh(self.attention_layer(lstm_out)) 
        
        # 2. Normalize scores into percentages (probabilities summing to 1.0)
        # alpha shape: (Batch, Seq_Len, 1)
        alpha = F.softmax(attention_scores, dim=1) 
        
        # 3. Multiply the LSTM outputs by their importance scores
        context_vector = lstm_out * alpha
        
        # 4. Compress the sequence into a single highly-focused representation
        # representation shape: (Batch, Hidden_Size*2)
        representation = torch.sum(context_vector, dim=1)
        
        # --- FINAL PREDICTION ---
        predicted_rr = self.fc(representation)
        
        # We use a flag here so we don't break our old training scripts yet!
        if return_attention:
            # Squeeze removes the extra dimension so it's a flat list of scores for the GUI
            return predicted_rr, alpha.squeeze(-1) 
            
        return predicted_rr

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("Testing Upgraded XAI Architecture...")
    
    model = AttentionBiLSTM()
    dummy_input = torch.randn(1, 1, 3750) # 30 seconds of simulated data
    
    # Test standard forward pass
    output = model(dummy_input)
    print(f"✅ Prediction Shape: {output.shape} (Should be [1, 1])")
    
    # Test XAI forward pass
    output, attention_weights = model(dummy_input, return_attention=True)
    print(f"✅ Attention Weights Shape: {attention_weights.shape} (Should be [1, 3750])")
    print("🚀 Phase 1 Complete: Model is now Explainable!")