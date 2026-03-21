import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Anomaly class definitions (shared by model, client, app)
# ─────────────────────────────────────────────────────────────
ANOMALY_CLASSES = {
    0: {"name": "Normal",           "rr_range": (12, 20),  "severity": "safe",     "color": "#22c55e"},
    1: {"name": "Bradypnea",        "rr_range": (8,  11),  "severity": "warning",  "color": "#eab308"},
    2: {"name": "Apnea",            "rr_range": (0,   7),  "severity": "critical", "color": "#ef4444"},
    3: {"name": "Tachypnea",        "rr_range": (21, 25),  "severity": "warning",  "color": "#f97316"},
    4: {"name": "Severe Tachypnea", "rr_range": (26, 999), "severity": "critical", "color": "#dc2626"},
}
NUM_ANOMALY_CLASSES = len(ANOMALY_CLASSES)

def rr_to_anomaly_label(rr: float) -> int:
    """
    Rule-based pseudo-labelling from RR value (breaths/min).
    Used to generate anomaly supervision from the regression target.
    """
    if   rr < 8:   return 2   # Apnea
    elif rr < 12:  return 1   # Bradypnea
    elif rr <= 20: return 0   # Normal
    elif rr <= 25: return 3   # Tachypnea
    else:          return 4   # Severe Tachypnea


class AttentionBiLSTM(nn.Module):
    """
    Multi-task Bidirectional LSTM with:
      • Attention mechanism        (XAI — Phase 1)
      • RR regression head         (original task)
      • Anomaly classification head (Phase 5 novelty)

    The two heads share the same BiLSTM encoder, so the anomaly
    classifier gets XAI attention for free and adds only ~4 KB of
    extra parameters — negligible for edge deployment.
    """
    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 num_layers=2,
                 output_size=1,
                 num_anomaly_classes=NUM_ANOMALY_CLASSES):

        super(AttentionBiLSTM, self).__init__()

        # ── Shared encoder ──────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # BiLSTM doubles hidden size
        enc_dim = hidden_size * 2

        # ── Attention layer ─────────────────────────────────────
        self.attention_layer = nn.Linear(enc_dim, 1)

        # ── Head 1: RR regression ───────────────────────────────
        self.fc = nn.Linear(enc_dim, output_size)

        # ── Head 2: Anomaly classification ──────────────────────
        # Small MLP — keeps edge footprint tiny
        self.anomaly_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_anomaly_classes),
        )

    # ── Shared feature extraction ────────────────────────────────
    def _encode(self, x):
        """
        x: (B, 1, L) → representation: (B, enc_dim), alpha: (B, L)
        """
        x = x.transpose(1, 2)                        # (B, L, 1)
        lstm_out, _ = self.lstm(x)                    # (B, L, enc_dim)

        scores = torch.tanh(self.attention_layer(lstm_out))   # (B, L, 1)
        alpha  = F.softmax(scores, dim=1)                     # (B, L, 1)

        context = lstm_out * alpha                            # (B, L, enc_dim)
        rep     = torch.sum(context, dim=1)                   # (B, enc_dim)

        return rep, alpha.squeeze(-1)                         # (B, enc_dim), (B, L)

    # ── Forward pass ─────────────────────────────────────────────
    def forward(self, x, return_attention=False, return_anomaly=False):
        """
        Args:
            x               : (B, 1, L) input signal
            return_attention: also return attention weights (B, L)
            return_anomaly  : also return anomaly logits   (B, C)

        Returns (depending on flags):
            rr_pred                              — always
            rr_pred, alpha                       — return_attention=True
            rr_pred, anomaly_logits              — return_anomaly=True
            rr_pred, alpha, anomaly_logits       — both True
        """
        rep, alpha = self._encode(x)

        rr_pred       = self.fc(rep)                   # (B, 1)
        anomaly_logits = self.anomaly_head(rep)        # (B, C)

        if return_attention and return_anomaly:
            return rr_pred, alpha, anomaly_logits
        if return_attention:
            return rr_pred, alpha
        if return_anomaly:
            return rr_pred, anomaly_logits
        return rr_pred


# ─────────────────────────────────────────────────────────────
# Test block
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Multi-Task AttentionBiLSTM...")
    model = AttentionBiLSTM()
    dummy = torch.randn(2, 1, 3750)   # batch=2, 30s at 125Hz

    rr = model(dummy)
    print(f"RR-only       : {rr.shape}  (expect [2,1])")

    rr, alpha = model(dummy, return_attention=True)
    print(f"+ attention   : {alpha.shape}  (expect [2,3750])")

    rr, logits = model(dummy, return_anomaly=True)
    print(f"+ anomaly     : {logits.shape}  (expect [2,5])")

    rr, alpha, logits = model(dummy, return_attention=True, return_anomaly=True)
    print(f"All outputs   : rr={rr.shape} alpha={alpha.shape} logits={logits.shape}")

    probs = torch.softmax(logits, dim=-1)
    pred  = torch.argmax(probs, dim=-1)
    print(f"Predicted class: {pred.tolist()}  (0=Normal,1=Brady,2=Apnea,3=Tachy,4=SevTachy)")
    print("Phase 5 complete: model is now a respiratory health monitor!")