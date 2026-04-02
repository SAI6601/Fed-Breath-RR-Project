"""
capnobase_loader.py -- CapnoBase IEEE TBME Benchmark loader for Fed-Breath

Loads .mat files (MATLAB v7.3 / HDF5) from the CapnoBase dataset
(42 patients, 300 Hz, 8 min each), extracts the PPG signal, resamples
to 125 Hz, bandpass-filters, and returns tensors in the exact same
format as BidmcDataset: (1, 3750) with float32.

The ground-truth respiratory rate (RR) label is the mean of the
CO2-derived instantaneous RR values provided by the dataset.

HDF5 layout (per file):
    signal/pleth/y           -> (1, 144001) float64   PPG waveform
    reference/rr/co2/y       -> (N, 1)      float64   instantaneous RR
    param/samplingrate/pleth -> (1, 1)      float64   native sample rate

Usage:
    from capnobase_loader import CapnoBaseDataset, mat_probe
    ds = CapnoBaseDataset()
    x, y = ds[0]   # x: (1, 3750), y: scalar RR

Requires: h5py, scipy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy.signal import butter, filtfilt, resample

# ─────────────────────────────────────────────────────────────
# Configuration -- matches BidmcDataset
# ─────────────────────────────────────────────────────────────
CAPNO_DIR         = os.path.join("data", "capnobase")
CAPNO_SAMPLE_RATE = 300       # CapnoBase native sample rate
TARGET_RATE       = 125       # Fed-Breath model expects 125 Hz
SEGMENT_SECONDS   = 30
SEGMENT_LENGTH    = TARGET_RATE * SEGMENT_SECONDS   # 3750


def mat_probe(path: str):
    """
    Print all groups and datasets in an HDF5 .mat file.
    Useful for figuring out the exact field names in CapnoBase files.
    """
    print(f"\n{'='*60}")
    print(f"  MAT PROBE (HDF5): {os.path.basename(path)}")
    print(f"{'='*60}")

    with h5py.File(path, 'r') as f:
        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [DATASET] {name}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  [GROUP]   {name}/")
        f.visititems(_visitor)

    print(f"{'='*60}\n")


def _find_ppg_signal(f: h5py.File):
    """
    Extract the PPG signal from the HDF5 structure.

    Known paths:
      - signal/pleth/y  (primary)
      - signal/ppg/y    (alternate)

    Returns the 1-D numpy array of the PPG signal, or None if not found.
    """
    for path in ('signal/pleth/y', 'signal/ppg/y',
                 'signal/PLETH/y', 'signal/PPG/y'):
        if path in f:
            arr = np.array(f[path], dtype=np.float64).ravel()
            if len(arr) > 1000:
                return arr

    # Fallback: walk signal group looking for a large 1-D dataset
    if 'signal' in f:
        for key in f['signal'].keys():
            obj = f['signal'][key]
            if isinstance(obj, h5py.Group) and 'y' in obj:
                arr = np.array(obj['y'], dtype=np.float64).ravel()
                if len(arr) > 1000:
                    return arr
            elif isinstance(obj, h5py.Dataset):
                arr = np.array(obj, dtype=np.float64).ravel()
                if len(arr) > 1000:
                    return arr

    return None


def _find_rr_label(f: h5py.File):
    """
    Extract mean respiratory rate (BrPM) from the HDF5 structure.

    The CapnoBase TBME benchmark stores CO2-derived instantaneous RR as:
        reference/rr/co2/y  ->  (N, 1) array of RR values in BrPM

    Returns a float (mean RR in breaths/min), or 0.0 if not found.
    """
    # Primary: reference/rr/co2/y (the RR values; /x is timestamps)
    for path in ('reference/rr/co2/y', 'reference/rr/y',
                 'labels/rr/y', 'labels/rr/co2/y'):
        if path in f:
            arr = np.array(f[path], dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                return float(np.mean(arr))

    # Fallback: SFresults/Fusion/y (smart-fusion RR estimates)
    if 'SFresults/Fusion/y' in f:
        arr = np.array(f['SFresults/Fusion/y'], dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            return float(np.mean(arr))

    return 0.0


def _find_sample_rate(f: h5py.File):
    """
    Extract the native PPG sample rate from the HDF5 metadata.
    Falls back to CAPNO_SAMPLE_RATE (300 Hz) if not found.
    """
    for path in ('param/samplingrate/pleth', 'param/samplingrate/ppg',
                 'param/Fs', 'param/fs'):
        if path in f:
            try:
                val = np.array(f[path]).ravel()
                return int(val[0])
            except (TypeError, ValueError, IndexError):
                pass

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
            with h5py.File(mat_path, 'r') as f:
                # -- 1. Extract PPG signal --
                raw_ppg = _find_ppg_signal(f)
                if raw_ppg is None:
                    print(f"[!!] No PPG signal found in {self.files[idx]}")
                    print(f"     Run: mat_probe('{mat_path}') to inspect")
                    return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

                # -- 2. Extract RR label --
                true_rr = _find_rr_label(f)

                # -- 3. Get native sample rate --
                native_sr = _find_sample_rate(f)

        except Exception as e:
            print(f"[!!] Error loading {self.files[idx]}: {e}")
            return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

        # -- 4. Resample to 125 Hz --
        if native_sr != TARGET_RATE:
            num_target_samples = int(len(raw_ppg) * TARGET_RATE / native_sr)
            ppg_resampled = resample(raw_ppg, num_target_samples)
        else:
            ppg_resampled = raw_ppg.copy()

        # -- 5. Scrub NaNs --
        ppg_resampled = np.nan_to_num(ppg_resampled, nan=0.0,
                                       posinf=0.0, neginf=0.0)

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
