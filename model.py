import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    """Rule-based pseudo-labelling from RR value (breaths/min)."""
    if   rr < 8:   return 2
    elif rr < 12:  return 1
    elif rr <= 20: return 0
    elif rr <= 25: return 3
    else:          return 4


class AttentionBiLSTM(nn.Module):
    """
    Multi-task Bidirectional LSTM with:
      • Attention mechanism          (XAI — Phase 1)
      • RR regression head           (original task)
      • Anomaly classification head  (Phase 5)
      • MC Dropout uncertainty       (Phase 6 — Academic credibility)

    The Dropout layer is placed on the representation vector so it
    applies to BOTH heads simultaneously during MC sampling, giving
    consistent uncertainty estimates across tasks.
    """
    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 num_layers=2,
                 output_size=1,
                 num_anomaly_classes=NUM_ANOMALY_CLASSES,
                 mc_dropout_p=0.3):   # increased from 0.1 for better CI calibration

        super(AttentionBiLSTM, self).__init__()

        enc_dim = hidden_size * 2

        # ── Shared encoder ───────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # ── Attention ────────────────────────────────────────────
        self.attention_layer = nn.Linear(enc_dim, 1)

        # ── MC Dropout on representation (shared across heads) ───
        # NOTE: Using nn.Dropout (not functional) so we can force
        # it active during inference via model.train() selectively.
        self.mc_dropout = nn.Dropout(p=mc_dropout_p)

        # ── Head 1: RR regression ────────────────────────────────
        self.fc = nn.Linear(enc_dim, output_size)

        # ── Head 2: Anomaly classification ──────────────────────
        self.anomaly_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_anomaly_classes),
        )

        self.mc_dropout_p = mc_dropout_p

    def _encode(self, x):
        """x: (B,1,L) → rep: (B, enc_dim), alpha: (B, L)"""
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        scores = torch.tanh(self.attention_layer(lstm_out))
        alpha  = F.softmax(scores, dim=1)
        rep    = torch.sum(lstm_out * alpha, dim=1)
        return rep, alpha.squeeze(-1)

    def forward(self, x, return_attention=False, return_anomaly=False):
        """
        Standard forward pass (Dropout inactive in eval mode).
        For uncertainty estimation use predict_with_uncertainty().
        """
        rep, alpha    = self._encode(x)
        rep           = self.mc_dropout(rep)       # no-op in eval mode
        rr_pred       = self.fc(rep)
        anomaly_logits = self.anomaly_head(rep)

        if return_attention and return_anomaly:
            return rr_pred, alpha, anomaly_logits
        if return_attention:
            return rr_pred, alpha
        if return_anomaly:
            return rr_pred, anomaly_logits
        return rr_pred

    # ── MC Dropout Uncertainty Quantification ────────────────────
    def predict_with_uncertainty(self, x, n_samples: int = 20, device=None):
        """
        Runs N stochastic forward passes with Dropout ACTIVE to
        estimate prediction uncertainty (epistemic).

        Args:
            x        : (B, 1, L) input tensor
            n_samples: number of MC samples (default 20, paper uses 50)
            device   : torch device (inferred from x if None)

        Returns dict:
            rr_mean   : (B,)  mean predicted RR across samples
            rr_std    : (B,)  std deviation — the uncertainty estimate
            rr_lower  : (B,)  mean - 1.96*std  (95% CI lower)
            rr_upper  : (B,)  mean - 1.96*std  (95% CI upper)
            ano_probs : (B,C) mean softmax probabilities across samples
            ano_std   : (B,C) std of softmax probs (uncertainty per class)
        """
        if device is None:
            device = x.device

        # Force Dropout ON for all MC passes (both self.mc_dropout
        # and the Dropout inside anomaly_head)
        self.train()

        rr_samples  = []
        ano_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                rep, _         = self._encode(x)
                rep            = self.mc_dropout(rep)
                rr_samples.append(self.fc(rep).squeeze(-1))          # (B,)
                ano_samples.append(
                    torch.softmax(self.anomaly_head(rep), dim=-1)     # (B,C)
                )

        # Restore eval mode
        self.eval()

        rr_stack  = torch.stack(rr_samples,  dim=0)   # (N, B)
        ano_stack = torch.stack(ano_samples, dim=0)   # (N, B, C)

        rr_mean  = rr_stack.mean(dim=0)               # (B,)
        rr_std   = rr_stack.std(dim=0)                # (B,)
        ano_mean = ano_stack.mean(dim=0)              # (B, C)
        ano_std  = ano_stack.std(dim=0)               # (B, C)

        return {
            "rr_mean":  rr_mean.cpu().numpy(),
            "rr_std":   rr_std.cpu().numpy(),
            "rr_lower": (rr_mean - 1.96 * rr_std).cpu().numpy(),
            "rr_upper": (rr_mean + 1.96 * rr_std).cpu().numpy(),
            "ano_probs": ano_mean.cpu().numpy(),
            "ano_std":   ano_std.cpu().numpy(),
        }


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Multi-Task AttentionBiLSTM with MC Dropout...")
    model = AttentionBiLSTM()
    model.eval()
    dummy = torch.randn(4, 1, 3750)

    rr = model(dummy)
    print(f"RR-only        : {rr.shape}")

    rr, alpha = model(dummy, return_attention=True)
    print(f"+ attention    : {alpha.shape}")

    rr, logits = model(dummy, return_anomaly=True)
    print(f"+ anomaly      : {logits.shape}")

    rr, alpha, logits = model(dummy, return_attention=True, return_anomaly=True)
    print(f"All outputs    : rr={rr.shape} alpha={alpha.shape} logits={logits.shape}")

    unc = model.predict_with_uncertainty(dummy, n_samples=20)
    print(f"\nMC Dropout (20 samples):")
    print(f"  rr_mean  : {unc['rr_mean'].round(3)}")
    print(f"  rr_std   : {unc['rr_std'].round(4)}  ← epistemic uncertainty")
    print(f"  95% CI   : [{unc['rr_lower'].round(2)}, {unc['rr_upper'].round(2)}]")
    print(f"  ano_probs: shape {unc['ano_probs'].shape}")
    print("All tests passed.")