import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# ANOMALY CLASSIFICATION CONFIG
# ─────────────────────────────────────────────────────────────
NUM_ANOMALY_CLASSES = 5

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
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=1, num_anomaly_classes=NUM_ANOMALY_CLASSES,
                 mc_dropout_p=0.3):
        """
        Multi-Task Attention-BiLSTM with MC Dropout uncertainty.

        Predicts:
          - Respiratory Rate (regression head)
          - Anomaly class (5-class classification head)

        Args:
            mc_dropout_p: Dropout probability for MC Dropout uncertainty
                          quantification. Applied during both training and
                          inference (when using predict_with_uncertainty).
                          Higher values -> wider confidence intervals.
        """
        super(AttentionBiLSTM, self).__init__()

        self.mc_dropout_p = mc_dropout_p

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

        # 3. MC Dropout layer (used for uncertainty quantification)
        self.mc_dropout = nn.Dropout(p=mc_dropout_p)

        # 4. TASK A: Respiratory Rate Regressor
        self.fc_rr = nn.Linear(hidden_size * 2, output_size)

        # 5. TASK B: Anomaly Classifier
        self.fc_anomaly = nn.Linear(hidden_size * 2, num_anomaly_classes)

        # 6. Temperature scaling parameter (for post-hoc calibration)
        #    Initialised to 1.0 (no change) -- optimised via calibrate_temperature()
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x, return_attention=False, return_anomaly=False):
        """
        Forward pass with flexible output signatures:

            model(x)                                    -> rr_pred
            model(x, return_attention=True)             -> rr_pred, alpha
            model(x, return_anomaly=True)               -> rr_pred, anomaly_logits
            model(x, return_attention=True,
                     return_anomaly=True)               -> rr_pred, alpha, anomaly_logits

        x shape: (Batch_Size, Features, Sequence_Length)
        """
        # Swap dimensions -> (Batch, Seq_Len, Features)
        x = x.transpose(1, 2)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # --- ATTENTION MECHANISM ---
        attention_scores = torch.tanh(self.attention_layer(lstm_out))
        alpha = F.softmax(attention_scores, dim=1)
        context_vector = lstm_out * alpha
        representation = torch.sum(context_vector, dim=1)

        # --- MC DROPOUT ---
        representation = self.mc_dropout(representation)

        # --- MULTI-TASK PREDICTIONS ---
        predicted_rr = self.fc_rr(representation)

        # Build return tuple based on flags
        if return_attention and return_anomaly:
            anomaly_logits = self.fc_anomaly(representation)
            return predicted_rr, alpha.squeeze(-1), anomaly_logits

        if return_anomaly:
            anomaly_logits = self.fc_anomaly(representation)
            return predicted_rr, anomaly_logits

        if return_attention:
            return predicted_rr, alpha.squeeze(-1)

        return predicted_rr

    def predict_with_uncertainty(self, x, n_samples=20):
        """
        MC Dropout uncertainty estimation.

        Runs n_samples stochastic forward passes with dropout enabled,
        then computes mean, std, and 95% confidence interval for the
        RR prediction.

        Args:
            x: Input tensor (Batch, 1, 3750)
            n_samples: Number of MC dropout samples (more = smoother)

        Returns:
            dict with keys:
                rr_mean  : (Batch,) mean predicted RR
                rr_std   : (Batch,) standard deviation
                rr_lower : (Batch,) lower bound of 95% CI
                rr_upper : (Batch,) upper bound of 95% CI
                rr_samples: (n_samples, Batch) raw predictions
        """
        was_training = self.training
        self.train()  # Enable dropout

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)  # (Batch, 1)
                samples.append(pred.squeeze(-1))  # (Batch,)

        if not was_training:
            self.eval()

        # Stack: (n_samples, Batch)
        samples_tensor = torch.stack(samples, dim=0)

        # Apply temperature scaling to the spread
        rr_mean = samples_tensor.mean(dim=0)
        rr_std = samples_tensor.std(dim=0) * self.temperature.item()

        rr_lower = rr_mean - 1.96 * rr_std
        rr_upper = rr_mean + 1.96 * rr_std

        return {
            "rr_mean": rr_mean,
            "rr_std": rr_std,
            "rr_lower": rr_lower,
            "rr_upper": rr_upper,
            "rr_samples": samples_tensor,
        }

    def calibrate_temperature(self, val_loader, device=None):
        """
        Post-hoc temperature scaling calibration (Guo et al., 2017).

        Optimises a single scalar temperature parameter to map raw MC
        Dropout std to calibrated confidence intervals, targeting 95%
        coverage on the validation set.

        This does NOT retrain the model -- only adjusts the temperature
        parameter used in predict_with_uncertainty().

        Args:
            val_loader: DataLoader with (x, y) pairs
            device: torch.device (defaults to CPU)
        """
        if device is None:
            device = torch.device("cpu")

        was_training = self.training
        self.train()  # Enable dropout for MC samples

        all_means, all_raw_stds, all_targets = [], [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)

                # Collect MC samples with temperature=1.0
                old_temp = self.temperature.item()
                self.temperature.data.fill_(1.0)

                unc = self.predict_with_uncertainty(inputs, n_samples=30)

                all_means.append(unc["rr_mean"].cpu())
                all_raw_stds.append(unc["rr_std"].cpu())
                all_targets.append(targets)

                self.temperature.data.fill_(old_temp)

        means = torch.cat(all_means)
        stds = torch.cat(all_raw_stds)
        targets = torch.cat(all_targets)

        # Binary search for optimal temperature that gives ~95% coverage
        best_temp = 1.0
        best_diff = float("inf")

        for t_candidate in [i * 0.1 for i in range(1, 51)]:
            lower = means - 1.96 * stds * t_candidate
            upper = means + 1.96 * stds * t_candidate
            coverage = float(((targets >= lower) & (targets <= upper)).float().mean() * 100)
            diff = abs(coverage - 95.0)
            if diff < best_diff:
                best_diff = diff
                best_temp = t_candidate

        self.temperature.data.fill_(best_temp)
        print(f"[Calibration] Temperature set to {best_temp:.1f} "
              f"(target: 95% coverage)")

        if not was_training:
            self.eval()


# ─────────────────────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Multi-Task XAI Architecture...")

    model = AttentionBiLSTM()
    dummy_input = torch.randn(1, 1, 3750)  # 30 seconds of simulated data

    # Test 1: basic forward
    rr = model(dummy_input)
    print(f"[OK] RR only: {rr.shape} (expected [1, 1])")

    # Test 2: with attention
    rr, attn = model(dummy_input, return_attention=True)
    print(f"[OK] RR + Attention: rr={rr.shape}, attn={attn.shape}")

    # Test 3: with anomaly (2 outputs)
    rr, anomaly = model(dummy_input, return_anomaly=True)
    print(f"[OK] RR + Anomaly: rr={rr.shape}, anomaly={anomaly.shape}")

    # Test 4: full multi-task (3 outputs)
    rr, attn, anomaly = model(dummy_input, return_attention=True, return_anomaly=True)
    print(f"[OK] Full: rr={rr.shape}, attn={attn.shape}, anomaly={anomaly.shape}")

    # Test 5: MC Dropout uncertainty
    unc = model.predict_with_uncertainty(dummy_input, n_samples=10)
    print(f"[OK] MC Dropout: mean={unc['rr_mean'].shape}, "
          f"std={unc['rr_std'].shape}, "
          f"CI=[{unc['rr_lower'].item():.2f}, {unc['rr_upper'].item():.2f}]")

    print("[OK] All forward pass signatures verified.")
    print(f"[OK] MC dropout_p = {model.mc_dropout_p}")
    print(f"[OK] Temperature = {model.temperature.item():.1f}")
    print("[OK] Model is ready.")