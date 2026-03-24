import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# ANOMALY CLASSIFICATION CONFIG
# ─────────────────────────────────────────────────────────────
ANOMALY_CLASSES = {
    0: {"name": "Normal", "severity": "safe", "color": "#22c55e"},
    1: {"name": "Bradypnea", "severity": "warning", "color": "#eab308"},
    2: {"name": "Apnea", "severity": "critical", "color": "#ef4444"},
    3: {"name": "Tachypnea", "severity": "warning", "color": "#f97316"},
    4: {"name": "Severe Tachypnea", "severity": "critical", "color": "#dc2626"}
}

def rr_to_anomaly_label(rr):
    """
    Converts a continuous Respiratory Rate (RR) into a discrete anomaly class ID 
    based on standard clinical ranges.
    """
    if rr < 8.0:
        return 2  # Apnea
    elif 8.0 <= rr < 12.0:
        return 1  # Bradypnea
    elif 12.0 <= rr <= 20.0:
        return 0  # Normal
    elif 20.0 < rr <= 25.0:
        return 3  # Tachypnea
    else:
        return 4  # Severe Tachypnea

# ─────────────────────────────────────────────────────────────
# CORE NEURAL NETWORK
# ─────────────────────────────────────────────────────────────
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, num_anomaly_classes=5):
        """
        Upgraded Multi-Task Attention-BiLSTM.
        Predicts both Respiratory Rate (Regression) and Anomalies (Classification).
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
        
        # 2. The Attention Spotlight
        self.attention_layer = nn.Linear(hidden_size * 2, 1)
        
        # 3. TASK A: Respiratory Rate Regressor
        self.fc_rr = nn.Linear(hidden_size * 2, output_size)
        
        # 4. TASK B: Anomaly Classifier
        self.fc_anomaly = nn.Linear(hidden_size * 2, num_anomaly_classes)

    def forward(self, x, return_attention=False, return_anomaly=False):
        """
        x shape: (Batch_Size, Features, Sequence_Length)
        """
        # Swap dimensions to match PyTorch LSTM expectations -> (Batch, Seq_Len, Features)
        x = x.transpose(1, 2) 
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # --- ATTENTION MECHANISM ---
        attention_scores = torch.tanh(self.attention_layer(lstm_out)) 
        alpha = F.softmax(attention_scores, dim=1) 
        context_vector = lstm_out * alpha
        representation = torch.sum(context_vector, dim=1)
        
        # --- MULTI-TASK PREDICTIONS ---
        predicted_rr = self.fc_rr(representation)
        
        if return_anomaly:
            # Output the raw logits for the 5 anomaly classes
            anomaly_logits = self.fc_anomaly(representation)
            return predicted_rr, alpha.squeeze(-1), anomaly_logits
            
        if return_attention:
            return predicted_rr, alpha.squeeze(-1) 
            
        return predicted_rr

# ─────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Multi-Task XAI Architecture...")
    
    model = AttentionBiLSTM()
    dummy_input = torch.randn(1, 1, 3750) # 30 seconds of simulated data
    
    # Test full multi-task forward pass
    rr, attn, anomalies = model(dummy_input, return_anomaly=True)
    print(f"✅ RR Shape: {rr.shape} (Should be [1, 1])")
    print(f"✅ Attention Shape: {attn.shape} (Should be [1, 3750])")
    print(f"✅ Anomaly Logits Shape: {anomalies.shape} (Should be [1, 5])")
    print("🚀 Model is ready for the new Dashboard!")