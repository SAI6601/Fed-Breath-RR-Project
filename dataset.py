import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
DATA_DIR = 'data/raw'
SAMPLE_RATE = 125
SEGMENT_SECONDS = 30  # We will cut data into 30-second chunks
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_SECONDS

class BidmcDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        """
        1. Finds all CSV files in the data folder.
        2. Prepares a list of valid files to train on.
        """
        self.files = [f for f in os.listdir(data_dir) if f.endswith('Signals.csv')]
        self.data_dir = data_dir
        
        print(f"✅ Found {len(self.files)} patient files in {data_dir}")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        The 'Magic' Method.
        PyTorch calls this when it asks for "Sample #5".
        It loads the file, cleans it, and returns the Tensors.
        """
        # 1. Identify file
        signal_file = self.files[idx]
        patient_id = signal_file.split('_')[1] # Extracts '01' from 'bidmc_01_Signals.csv'
        
        # 2. Load Signal Data
        sig_path = os.path.join(self.data_dir, signal_file)
        df_sig = pd.read_csv(sig_path)
        
        # Smart column finder
        ppg_col = [c for c in df_sig.columns if 'PLETH' in c.upper()][0]
        raw_ppg = df_sig[ppg_col].values

        # 3. Load Label Data (True Respiratory Rate)
        numerics_file = f'bidmc_{patient_id}_Numerics.csv'
        num_path = os.path.join(self.data_dir, numerics_file)
        
        if not os.path.exists(num_path):
            true_rr = 0.0
        else:
            df_num = pd.read_csv(num_path)
            resp_col = [c for c in df_num.columns if 'RESP' in c.upper()][0]
            true_rr = df_num[resp_col].mean()

        # 4. Preprocessing (Bandpass Filter)
        clean_ppg = self.preprocess(raw_ppg)

        # 5. Cropping (Ensure fixed size for Neural Network)
        if len(clean_ppg) > SEGMENT_LENGTH:
            clean_ppg = clean_ppg[:SEGMENT_LENGTH]
        else:
            clean_ppg = np.pad(clean_ppg, (0, SEGMENT_LENGTH - len(clean_ppg)))

        # 6. Convert to PyTorch Tensors
        # --- FIX IS HERE: Added .copy() to ensure positive strides ---
        x_tensor = torch.tensor(clean_ppg.copy(), dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(true_rr, dtype=torch.float32)

        return x_tensor, y_tensor

    def preprocess(self, data):
        """Applies 0.1-0.5Hz Bandpass Filter"""
        nyquist = 0.5 * SAMPLE_RATE
        low = 0.1 / nyquist
        high = 0.5 / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("Testing Dataset Loader...")
    
    dataset = BidmcDataset()
    
    if len(dataset) == 0:
        print("❌ Error: No files found. Check your 'data/raw' folder.")
        exit()

    # Get the first sample (Patient 01)
    try:
        x, y = dataset[0]
        print(f"\nSample 01 Loaded Successfully!")
        print(f"Shape of Input (PPG): {x.shape} (Should be [1, 3750])")
        print(f"Label (True RR): {y.item():.2f} breaths/min")
        print("✅ Data is clean and ready for AI.")
        
    except Exception as e:
        print(f"❌ Error loading sample: {e}")