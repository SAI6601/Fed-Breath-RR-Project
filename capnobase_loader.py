"""
capnobase_loader.py -- CapnoBase IEEE TBME Benchmark loader for Fed-Breath

Loads .mat files from the CapnoBase dataset (42 patients, 300 Hz, 8 min each),
extracts the PPG signal, resamples to 125 Hz, bandpass-filters, and returns
tensors in the exact same format as BidmcDataset: (1, 3750) with float32.

The ground-truth respiratory rate (RR) label is the mean of the CO2-derived
instantaneous RR values provided by the dataset.

Usage:
    from capnobase_loader import CapnoBaseDataset, mat_probe
    ds = CapnoBaseDataset()
    x, y = ds[0]   # x: (1, 3750), y: scalar RR

Requires: scipy  (for .mat loading + resampling + filtering)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample

# ─────────────────────────────────────────────────────────────
# Configuration -- matches BidmcDataset
# ─────────────────────────────────────────────────────────────
CAPNO_DIR        = os.path.join("data", "capnobase")
CAPNO_SAMPLE_RATE = 300       # CapnoBase native sample rate
TARGET_RATE       = 125       # Fed-Breath model expects 125 Hz
SEGMENT_SECONDS   = 30
SEGMENT_LENGTH    = TARGET_RATE * SEGMENT_SECONDS   # 3750


def mat_probe(path: str):
    """
    Print all top-level keys and nested structures in a .mat file.
    Useful for figuring out the exact field names in CapnoBase files
    since naming can vary between versions.
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    print(f"\n{'='*60}")
    print(f"  MAT PROBE: {os.path.basename(path)}")
    print(f"{'='*60}")

    for key in sorted(mat.keys()):
        if key.startswith("__"):
            continue
        val = mat[key]
        _print_nested(key, val, indent=0)

    print(f"{'='*60}\n")


def _print_nested(name, obj, indent=0):
    """Recursively print structure contents."""
    prefix = "  " * indent
    if hasattr(obj, '_fieldnames'):
        # MATLAB struct
        print(f"{prefix}[STRUCT] {name}  (fields: {obj._fieldnames})")
        for field in obj._fieldnames:
            _print_nested(field, getattr(obj, field), indent + 1)
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}[ARRAY] {name}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, (int, float, str)):
        print(f"{prefix}[VALUE] {name}  = {obj}")
    else:
        print(f"{prefix}[??] {name}  type={type(obj).__name__}")


def _find_ppg_signal(mat: dict):
    """
    Auto-discover the PPG signal array from the .mat structure.

    CapnoBase stores signals under various keys depending on the version:
      - signal.pleth.y
      - signal.ppg.y
      - data.ppg
      - ppg  (top-level)

    Returns the 1-D numpy array of the PPG signal, or None if not found.
    """
    # Strategy 1: signal struct -> pleth or ppg sub-struct -> y
    if "signal" in mat:
        sig_struct = mat["signal"]
        if hasattr(sig_struct, '_fieldnames'):
            for candidate in ("pleth", "ppg", "PLETH", "PPG"):
                if candidate in sig_struct._fieldnames:
                    sub = getattr(sig_struct, candidate)
                    if hasattr(sub, '_fieldnames') and "y" in sub._fieldnames:
                        return np.asarray(getattr(sub, "y"), dtype=np.float64).ravel()
                    elif isinstance(sub, np.ndarray):
                        return sub.astype(np.float64).ravel()

    # Strategy 2: data struct -> ppg
    if "data" in mat:
        data_struct = mat["data"]
        if hasattr(data_struct, '_fieldnames'):
            for candidate in ("ppg", "pleth", "PPG", "PLETH"):
                if candidate in data_struct._fieldnames:
                    arr = getattr(data_struct, candidate)
                    return np.asarray(arr, dtype=np.float64).ravel()

    # Strategy 3: top-level keys
    for key in ("ppg", "pleth", "PPG", "PLETH", "signal"):
        if key in mat and isinstance(mat[key], np.ndarray):
            arr = mat[key].astype(np.float64).ravel()
            if len(arr) > 1000:   # must be a signal, not metadata
                return arr

    return None


def _find_rr_label(mat: dict):
    """
    Extract mean respiratory rate (BrPM) from the .mat structure.

    CapnoBase provides CO2-derived instantaneous RR under:
      - reference.rr.co2  (scalar or array)
      - reference.rr      (if no sub-keys)
      - labels.rr         (fallback)

    Returns a float (mean RR in breaths/min), or 0.0 if not found.
    """
    # Strategy 1: reference struct
    if "reference" in mat:
        ref = mat["reference"]
        if hasattr(ref, '_fieldnames'):
            # reference.rr -> may be struct with co2, or array directly
            if "rr" in ref._fieldnames:
                rr_obj = getattr(ref, "rr")
                if hasattr(rr_obj, '_fieldnames'):
                    # reference.rr.co2
                    for sub_key in ("co2", "CO2", "x", "y", "value"):
                        if sub_key in rr_obj._fieldnames:
                            arr = np.asarray(getattr(rr_obj, sub_key), dtype=np.float64).ravel()
                            arr = arr[np.isfinite(arr)]
                            if len(arr) > 0:
                                return float(np.mean(arr))
                elif isinstance(rr_obj, np.ndarray):
                    arr = rr_obj.astype(np.float64).ravel()
                    arr = arr[np.isfinite(arr)]
                    if len(arr) > 0:
                        return float(np.mean(arr))
                elif isinstance(rr_obj, (int, float)):
                    return float(rr_obj)

    # Strategy 2: labels struct
    if "labels" in mat:
        labels = mat["labels"]
        if hasattr(labels, '_fieldnames') and "rr" in labels._fieldnames:
            rr_obj = getattr(labels, "rr")
            if isinstance(rr_obj, np.ndarray):
                arr = rr_obj.astype(np.float64).ravel()
                arr = arr[np.isfinite(arr)]
                if len(arr) > 0:
                    return float(np.mean(arr))

    # Strategy 3: param struct for breathing rate
    if "param" in mat:
        param = mat["param"]
        if hasattr(param, '_fieldnames'):
            for key in ("rr", "RR", "resp_rate", "breathing_rate"):
                if key in param._fieldnames:
                    val = getattr(param, key)
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass

    return 0.0


def _find_sample_rate(mat: dict):
    """Extract the native sample rate from the .mat metadata."""
    if "param" in mat:
        param = mat["param"]
        if hasattr(param, '_fieldnames'):
            for key in ("samplingrate", "Fs", "fs", "sr", "SamplingRate",
                        "sampling_rate", "samplerate"):
                if key in param._fieldnames:
                    try:
                        return int(getattr(param, key))
                    except (TypeError, ValueError):
                        pass

    # Fallback: CapnoBase default
    return CAPNO_SAMPLE_RATE


# ─────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────
class CapnoBaseDataset(Dataset):
    """
    PyTorch Dataset for CapnoBase IEEE TBME Benchmark.

    Each item returns:
        x: (1, 3750) float32 tensor -- 30s PPG at 125 Hz, bandpass + z-normalised
        y: float32 scalar           -- mean respiratory rate (BrPM)
    """

    def __init__(self, data_dir: str = CAPNO_DIR):
        self.data_dir = data_dir

        if not os.path.isdir(data_dir):
            print(f"[WARN] CapnoBase directory not found: {data_dir}")
            print(f"       Create it and place .mat files inside.")
            self.files = []
            return

        # Discover all .mat files (signal files)
        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".mat")
        ])

        print(f"[CapnoBase] {len(self.files)} patients in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat_path = os.path.join(self.data_dir, self.files[idx])

        try:
            mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"[!!] Error loading {self.files[idx]}: {e}")
            return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

        # -- 1. Extract PPG signal --
        raw_ppg = _find_ppg_signal(mat)
        if raw_ppg is None:
            print(f"[!!] No PPG signal found in {self.files[idx]}")
            print(f"     Run: mat_probe('{mat_path}') to inspect the file structure")
            return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

        # -- 2. Extract RR label --
        true_rr = _find_rr_label(mat)

        # -- 3. Get native sample rate --
        native_sr = _find_sample_rate(mat)

        # -- 4. Resample to 125 Hz --
        if native_sr != TARGET_RATE:
            num_target_samples = int(len(raw_ppg) * TARGET_RATE / native_sr)
            ppg_resampled = resample(raw_ppg, num_target_samples)
        else:
            ppg_resampled = raw_ppg.copy()

        # -- 5. Scrub NaNs --
        ppg_resampled = np.nan_to_num(ppg_resampled, nan=0.0)

        # -- 6. Bandpass filter (0.1-0.5 Hz, matches BIDMC) --
        clean_ppg = self._bandpass(ppg_resampled)

        # -- 7. Crop/pad to 3750 samples --
        if len(clean_ppg) > SEGMENT_LENGTH:
            clean_ppg = clean_ppg[:SEGMENT_LENGTH]
        else:
            clean_ppg = np.pad(clean_ppg, (0, SEGMENT_LENGTH - len(clean_ppg)))

        # -- 8. Z-score normalisation --
        std = np.std(clean_ppg)
        clean_ppg = (clean_ppg - np.mean(clean_ppg)) / (std + 1e-6)

        # -- 9. Convert to tensor --
        x = torch.tensor(clean_ppg.copy(), dtype=torch.float32).unsqueeze(0)

        # Final NaN safety check
        if torch.isnan(x).any():
            x = torch.zeros_like(x)

        y = torch.tensor(true_rr, dtype=torch.float32)

        return x, y

    @staticmethod
    def _bandpass(data, lowcut=0.1, highcut=0.5, fs=TARGET_RATE, order=4):
        """Bandpass filter to isolate breathing frequencies (same as BIDMC pipeline)."""
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        return filtfilt(b, a, data)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = CapnoBaseDataset()
    if len(ds) == 0:
        print("\n[WARN] No .mat files found. Place CapnoBase files in data/capnobase/")
    else:
        print(f"\nLoading first sample...")
        x, y = ds[0]
        print(f"  x shape  : {x.shape}   (expected: [1, 3750])")
        print(f"  y (RR)   : {y.item():.2f} BrPM")
        print(f"  x range  : [{x.min():.3f}, {x.max():.3f}]")
        print(f"  NaN check: {'PASS' if not torch.isnan(x).any() else 'FAIL'}")

        # Probe first file
        mat_probe(os.path.join(ds.data_dir, ds.files[0]))
