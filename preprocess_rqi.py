import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os

# --- CONFIGURATION ---
# We use the filename you verified yesterday
FILE_PATH = 'data/raw/bidmc_01_Signals.csv' 
SAMPLE_RATE = 125  # BIDMC data is 125Hz

def load_data(path):
    """Loads CSV and finds the correct PPG column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    # Smart search for the PPG column (handles ' PLETH', 'PLETH', etc.)
    ppg_col = [c for c in df.columns if 'PLETH' in c.upper().strip()][0]
    return df[ppg_col].values

def bandpass_filter(data, lowcut=0.1, highcut=0.5, fs=125, order=4):
    """
    Isolates breathing frequencies (6-30 breaths/min).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_rqi(respiratory_signal, fs=125):
    """
    Calculates the Respiratory Quality Index (RQI).
    Based on the consistency of breath-to-breath intervals.
    """
    # 1. Detect Peaks (Breaths)
    # distance=fs*2 means we expect breaths to be at least 2 seconds apart (max 30 bpm)
    peaks, _ = find_peaks(respiratory_signal, distance=fs*1.5, prominence=0.1)
    
    if len(peaks) < 2:
        return 0.0, peaks # Too few breaths to judge quality
    
    # 2. Calculate time intervals between breaths (in seconds)
    intervals = np.diff(peaks) / fs
    
    # 3. Calculate Coefficient of Variation (CV)
    # Formula: Standard Deviation / Mean
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    if mean_interval == 0:
        return 0.0, peaks
        
    cv = std_interval / mean_interval
    
    # 4. Convert to Quality Score (0 to 1)
    # If CV is 0 (perfect rhythm), RQI is 1.
    # If CV is high (erratic), RQI drops.
    rqi = max(0, 1 - cv)
    
    return rqi, peaks

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    print("1. Loading Data...")
    raw_ppg = load_data(FILE_PATH)
    
    # We only analyze the first 60 seconds (7500 samples) for this test
    segment_duration = 60 
    segment = raw_ppg[:SAMPLE_RATE * segment_duration]
    
    print("2. Preprocessing (Bandpass Filter 0.1-0.5Hz)...")
    clean_resp = bandpass_filter(segment)
    
    print("3. Calculating RQI (Quality Score)...")
    rqi_score, peaks = calculate_rqi(clean_resp)
    
    print(f"\n===========================")
    print(f"📊 RQI SCORE: {rqi_score:.4f}")
    print(f"===========================")
    if rqi_score > 0.8:
        print("✅ Excellent Quality: Rhythm is consistent.")
    elif rqi_score > 0.5:
        print("⚠️ Moderate Quality: Some irregularity.")
    else:
        print("❌ Poor Quality: Too noisy to trust.")

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    
    # Plot Filtered Signal
    plt.plot(np.arange(len(clean_resp))/SAMPLE_RATE, clean_resp, color='green', label='Filtered Breathing Signal')
    
    # Plot Detected Peaks (Red dots)
    plt.plot(peaks/SAMPLE_RATE, clean_resp[peaks], "ro", label='Detected Breaths')
    
    plt.title(f"Extracted Breathing Signal (RQI Score: {rqi_score:.2f})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()