import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample

# --- CONFIGURATION ---
DATA_DIR = 'data/raw'
SAMPLE_RATE = 125
SEGMENT_SECONDS = 30
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS  # 3750


def _preprocess_ppg(raw, src_fs, target_fs=SAMPLE_RATE,
                    lowcut=0.1, highcut=0.5, order=4,
                    segment_length=SEGMENT_LENGTH):
    """
    Shared preprocessing pipeline for all dataset loaders.

    Steps:
        1. Resample to target_fs (125 Hz)
        2. Bandpass filter 0.1-0.5 Hz (breathing band)
        3. Crop/pad to segment_length (3750 samples = 30 s)
        4. Z-score normalisation

    Args:
        raw: 1-D numpy array of raw PPG signal
        src_fs: source sample rate (e.g. 125 for BIDMC, 300 for CapnoBase)
        target_fs: target sample rate (125 Hz)
        lowcut: bandpass low cutoff (Hz)
        highcut: bandpass high cutoff (Hz)
        order: Butterworth filter order
        segment_length: output length in samples

    Returns:
        torch.Tensor of shape (1, segment_length), float32, z-normalised
    """
    raw = np.asarray(raw, dtype=np.float64)

    # 1. Resample if needed
    if src_fs != target_fs:
        num_samples = int(len(raw) * target_fs / src_fs)
        raw = resample(raw, num_samples)

    # 2. Scrub NaNs/Infs
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Bandpass filter
    nyq = 0.5 * target_fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = filtfilt(b, a, raw)

    # 4. Crop/pad to segment_length
    if len(filtered) > segment_length:
        filtered = filtered[:segment_length]
    else:
        filtered = np.pad(filtered, (0, segment_length - len(filtered)))

    # 5. Z-score normalisation
    std = np.std(filtered)
    filtered = (filtered - np.mean(filtered)) / (std + 1e-6)

    # 6. Convert to tensor
    x = torch.tensor(filtered.copy(), dtype=torch.float32).unsqueeze(0)

    # Final NaN safety check
    if torch.isnan(x).any():
        x = torch.zeros_like(x)

    return x


class BidmcDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.files = [f for f in os.listdir(data_dir) if f.endswith('Signals.csv')]
        self.data_dir = data_dir
        print(f"[OK] Found {len(self.files)} patient files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Load Data
        signal_file = self.files[idx]
        patient_id = signal_file.split('_')[1]
        sig_path = os.path.join(self.data_dir, signal_file)

        try:
            # Memory optimization: find signal column first without loading whole file
            header = pd.read_csv(sig_path, nrows=0).columns
            ppg_col = [c for c in header if 'PLETH' in c.upper()][0]
            df_sig = pd.read_csv(sig_path, usecols=[ppg_col])
            raw_ppg = df_sig[ppg_col].values.astype(np.float32)

            # Load Labels: find resp column first
            numerics_file = f'bidmc_{patient_id}_Numerics.csv'
            num_path = os.path.join(self.data_dir, numerics_file)
            if os.path.exists(num_path):
                num_header = pd.read_csv(num_path, nrows=0).columns
                resp_col = [c for c in num_header if 'RESP' in c.upper()][0]
                df_num = pd.read_csv(num_path, usecols=[resp_col])
                true_rr = df_num[resp_col].mean()
            else:
                true_rr = 0.0

        except Exception as e:
            print(f"[!!] Error loading {signal_file}: {e}")
            return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

        # 2. Preprocess using shared pipeline (BIDMC is already 125 Hz)
        x_tensor = _preprocess_ppg(raw_ppg, src_fs=SAMPLE_RATE)

        y_tensor = torch.tensor(true_rr, dtype=torch.float32)

        return x_tensor, y_tensor


# ─────────────────────────────────────────────────────────────
# CapnoBase Dataset (imported from dedicated loader)
# ─────────────────────────────────────────────────────────────
try:
    from capnobase_loader import CapnoBaseDataset
except ImportError:
    class CapnoBaseDataset(Dataset):
        def __init__(self): self.files = []
        def __len__(self): return 0

# ─────────────────────────────────────────────────────────────
# Apnea-ECG Dataset (Stub for Phase 5)
# ─────────────────────────────────────────────────────────────
class ApneaECGDataset(Dataset):
    """
    Stub for the PhysioNet Apnea-ECG dataset.
    Requires downloading .dat files to data/apnea/
    """
    def __init__(self, data_dir='data/apnea'):
        self.data_dir = data_dir
        self.files = []
        if os.path.isdir(data_dir):
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
        else:
            print(f"[WARN] Apnea-ECG directory not found: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Dummy return until properly loaded
        return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

# ─────────────────────────────────────────────────────────────
# Combined Dataset & Utilities
# ─────────────────────────────────────────────────────────────
from torch.utils.data import ConcatDataset

class CombinedDataset(ConcatDataset):
    """Wraps Pytorch ConcatDataset to instantiate requested datasets automatically."""
    def __init__(self, dataset_names):
        datasets = []
        ds_map = {
            "bidmc": BidmcDataset,
            "capnobase": CapnoBaseDataset,
            "apnea": ApneaECGDataset
        }
        for name in dataset_names:
            name = name.lower()
            if name in ds_map:
                ds = ds_map[name]()
                if len(ds) > 0:
                    datasets.append(ds)
        super().__init__(datasets)

def available_datasets():
    """Returns a dictionary mapping dataset names to their availability status."""
    status = {}
    
    # Check BIDMC
    bidmc_files = len([f for f in os.listdir(DATA_DIR) if f.endswith('Signals.csv')]) if os.path.isdir(DATA_DIR) else 0
    status["bidmc"] = {"ready": bidmc_files > 0, "files": bidmc_files}
    
    # Check CapnoBase
    capno_dir = os.path.join("data", "capnobase")
    capno_files = len([f for f in os.listdir(capno_dir) if f.endswith('.mat')]) if os.path.isdir(capno_dir) else 0
    status["capnobase"] = {"ready": capno_files > 0, "files": capno_files}
    
    # Check Apnea-ECG
    apnea_dir = os.path.join("data", "apnea")
    apnea_files = len([f for f in os.listdir(apnea_dir) if f.endswith('.dat')]) if os.path.isdir(apnea_dir) else 0
    status["apnea"] = {"ready": apnea_files > 0, "files": apnea_files}
    
    return status


if __name__ == "__main__":
    print("--- BIDMC Dataset Test ---")
    ds = BidmcDataset()
    print(f"  Patients: {len(ds)}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"  x shape: {x.shape}  y (RR): {y.item():.2f} BrPM")