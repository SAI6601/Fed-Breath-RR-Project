"""
test_capnobase.py -- Diagnostic test for CapnoBase dataset integration

Run after placing .mat files in data/capnobase/:
    python test_capnobase.py

Tests:
  1. Probe a .mat file to show its internal structure
  2. Load the full CapnoBaseDataset
  3. Verify tensor shapes and values
  4. Feed a sample through AttentionBiLSTM to confirm compatibility
  5. Print RR label distribution (should span <10 to >40 BrPM)
"""

import os
import sys
import torch
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. Probe .mat structure
# ─────────────────────────────────────────────────────────────
def test_probe():
    from capnobase_loader import mat_probe, CAPNO_DIR

    files = [f for f in os.listdir(CAPNO_DIR) if f.endswith(".mat")] if os.path.isdir(CAPNO_DIR) else []
    if not files:
        print("[SKIP] TEST 1: No .mat files found in data/capnobase/")
        return False

    path = os.path.join(CAPNO_DIR, files[0])
    print(f"\n[TEST 1] Probing {files[0]}...")
    mat_probe(path)
    print("[OK] TEST 1 PASS: .mat probe completed\n")
    return True


# ─────────────────────────────────────────────────────────────
# 2. Load full dataset
# ─────────────────────────────────────────────────────────────
def test_load_dataset():
    from capnobase_loader import CapnoBaseDataset

    print("[TEST 2] Loading CapnoBaseDataset...")
    ds = CapnoBaseDataset()

    if len(ds) == 0:
        print("[!!] TEST 2 FAIL: Dataset is empty")
        return None

    print(f"[OK] TEST 2 PASS: Loaded {len(ds)} patients\n")
    return ds


# ─────────────────────────────────────────────────────────────
# 3. Verify tensor shapes and values
# ─────────────────────────────────────────────────────────────
def test_tensors(ds):
    print("[TEST 3] Checking tensor shapes and values...")
    errors = 0
    rr_values = []

    for i in range(len(ds)):
        x, y = ds[i]
        rr_values.append(y.item())

        if x.shape != (1, 3750):
            print(f"  [!!] Sample {i}: x shape = {x.shape} (expected [1, 3750])")
            errors += 1

        if torch.isnan(x).any():
            print(f"  [!!] Sample {i}: contains NaN values")
            errors += 1

        if torch.isinf(x).any():
            print(f"  [!!] Sample {i}: contains Inf values")
            errors += 1

    rr_arr = np.array(rr_values)

    print(f"\n  Samples checked : {len(ds)}")
    print(f"  Shape errors    : {errors}")
    print(f"  RR range        : {rr_arr.min():.1f} -- {rr_arr.max():.1f} BrPM")
    print(f"  RR mean +/- std : {rr_arr.mean():.1f} +/- {rr_arr.std():.1f} BrPM")
    print(f"  RR = 0 (missing): {np.sum(rr_arr == 0.0)} files")

    # Clinical range check
    has_low  = np.any(rr_arr < 12)
    has_high = np.any(rr_arr > 25)
    print(f"  Has bradypnea   : {'YES' if has_low else 'NO'}")
    print(f"  Has tachypnea   : {'YES' if has_high else 'NO'}")

    if errors == 0:
        print(f"\n[OK] TEST 3 PASS: All {len(ds)} samples valid\n")
    else:
        print(f"\n[WARN] TEST 3: {errors} errors found\n")

    return rr_values


# ─────────────────────────────────────────────────────────────
# 4. Model compatibility
# ─────────────────────────────────────────────────────────────
def test_model_compat(ds):
    from model import AttentionBiLSTM

    print("[TEST 4] Model forward-pass compatibility...")

    model = AttentionBiLSTM()
    model.eval()

    x, y = ds[0]
    x_batch = x.unsqueeze(0)   # (1, 1, 3750)

    with torch.no_grad():
        rr_pred, alpha, anomaly_logits = model(
            x_batch, return_attention=True, return_anomaly=True
        )

    print(f"  Input shape      : {x_batch.shape}")
    print(f"  RR prediction    : {rr_pred.item():.2f} BrPM  (true: {y.item():.2f})")
    print(f"  Attention shape  : {alpha.shape}  (expected: [1, 3750])")
    print(f"  Anomaly logits   : {anomaly_logits.shape}  (expected: [1, 5])")
    print(f"  Anomaly class    : {torch.argmax(anomaly_logits, dim=1).item()}")

    ok = (rr_pred.shape == (1, 1) and
          alpha.shape == (1, 3750) and
          anomaly_logits.shape == (1, 5) and
          not torch.isnan(rr_pred).any())

    if ok:
        print(f"\n[OK] TEST 4 PASS: Model fully compatible\n")
    else:
        print(f"\n[!!] TEST 4 FAIL: Shape mismatch or NaN output\n")


# ─────────────────────────────────────────────────────────────
# 5. Combined dataset test
# ─────────────────────────────────────────────────────────────
def test_combined():
    from dataset import get_combined_dataset

    print("[TEST 5] Combined dataset (BIDMC + CapnoBase)...")
    combined = get_combined_dataset()
    print(f"  Total samples: {len(combined)}")

    # Spot-check last sample (should be from CapnoBase partition)
    x, y = combined[len(combined) - 1]
    print(f"  Last sample shape: {x.shape}, RR: {y.item():.2f} BrPM")
    print(f"\n[OK] TEST 5 PASS: Combined dataset works\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FED-BREATH: CapnoBase Compatibility Test Suite")
    print("=" * 60)

    # Test 1: Probe
    if not test_probe():
        print("\n[WARN] Place CapnoBase .mat files in data/capnobase/ and re-run.")
        sys.exit(1)

    # Test 2: Load
    ds = test_load_dataset()
    if ds is None:
        sys.exit(1)

    # Test 3: Tensors
    test_tensors(ds)

    # Test 4: Model
    test_model_compat(ds)

    # Test 5: Combined
    test_combined()

    print("=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
