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
            df_sig = pd.read_csv(sig_path)
            ppg_col = [c for c in df_sig.columns if 'PLETH' in c.upper()][0]
            raw_ppg = df_sig[ppg_col].values

            # Load Labels
            numerics_file = f'bidmc_{patient_id}_Numerics.csv'
            num_path = os.path.join(self.data_dir, numerics_file)
            if os.path.exists(num_path):
                df_num = pd.read_csv(num_path)
                resp_col = [c for c in df_num.columns if 'RESP' in c.upper()][0]
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
# Combined Dataset (BIDMC + CapnoBase)
# ─────────────────────────────────────────────────────────────
def get_combined_dataset(include_capnobase=True):
    """
    Returns a ConcatDataset wrapping BIDMC and (optionally) CapnoBase.
    Falls back to BIDMC-only if CapnoBase directory is empty or missing.
    """
    from torch.utils.data import ConcatDataset

    datasets = [BidmcDataset()]

    if include_capnobase:
        try:
            from capnobase_loader import CapnoBaseDataset
            capno = CapnoBaseDataset()
            if len(capno) > 0:
                datasets.append(capno)
                print(f"[OK] Combined dataset: {len(datasets[0])} BIDMC "
                      f"+ {len(capno)} CapnoBase = "
                      f"{len(datasets[0]) + len(capno)} total")
            else:
                print(f"[WARN] CapnoBase has 0 files -- using BIDMC only")
        except ImportError:
            print(f"[WARN] capnobase_loader not found -- using BIDMC only")

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


if __name__ == "__main__":
    print("--- BIDMC Dataset Test ---")
    ds = BidmcDataset()
    print(f"  Patients: {len(ds)}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"  x shape: {x.shape}  y (RR): {y.item():.2f} BrPM")