import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
DATA_DIR = 'data/raw'
SAMPLE_RATE = 125
SEGMENT_SECONDS = 30
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS

class BidmcDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.files = [f for f in os.listdir(data_dir) if f.endswith('Signals.csv')]
        self.data_dir = data_dir
        print(f"✅ Found {len(self.files)} patient files in {data_dir}")

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
            print(f"Error loading {signal_file}: {e}")
            return torch.zeros(1, SEGMENT_LENGTH), torch.tensor(0.0)

        # 2. Preprocessing
        clean_ppg = self.preprocess(raw_ppg)

        # 3. Cropping/Padding
        if len(clean_ppg) > SEGMENT_LENGTH:
            clean_ppg = clean_ppg[:SEGMENT_LENGTH]
        else:
            clean_ppg = np.pad(clean_ppg, (0, SEGMENT_LENGTH - len(clean_ppg)))

        # --- CRITICAL FIX: Z-Score Normalization ---
        # Subtract mean, divide by std deviation. Adds 1e-6 to avoid divide-by-zero.
        clean_ppg = (clean_ppg - np.mean(clean_ppg)) / (np.std(clean_ppg) + 1e-6)

        # 4. Convert to Tensor
        x_tensor = torch.tensor(clean_ppg.copy(), dtype=torch.float32).unsqueeze(0)
        
        # Check for NaNs one last time
        if torch.isnan(x_tensor).any():
            x_tensor = torch.zeros_like(x_tensor)
            
        y_tensor = torch.tensor(true_rr, dtype=torch.float32)

        return x_tensor, y_tensor

    def preprocess(self, data):
        nyquist = 0.5 * SAMPLE_RATE
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)