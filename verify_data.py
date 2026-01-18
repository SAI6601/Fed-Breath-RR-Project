import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CORRECTED FILENAME HERE ---
# Based on your error log, the file is named 'bidmc_01_Signals.csv'
file_path = 'data/raw/bidmc_01_Signals.csv'

# 1. Check if data exists
if not os.path.exists(file_path):
    print(f"❌ Error: File not found at {file_path}")
    if os.path.exists('data/raw'):
        print(f"Files found: {os.listdir('data/raw')[:5]}")
    exit()

# 2. Load data
print(f"✅ Found file! Loading data from {file_path}...")
try:
    df = pd.read_csv(file_path)
    print("Columns found:", df.columns.tolist())
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit()

# 3. Plot signal
print("Generating plot... (Close the popup window to finish)")
plt.figure(figsize=(12, 6))

# Note: The column names might also be slightly different in this version.
# We usually look for ' PLETH' or 'PLETH' (sometimes there is a leading space).
# This logic handles both:
ppg_col = [c for c in df.columns if 'PLETH' in c.upper().strip()][0]
resp_col = [c for c in df.columns if 'RESP' in c.upper().strip()][0]

# Plot PPG
plt.subplot(2, 1, 1)
plt.plot(df[ppg_col][:500], color='red', label='PPG Signal') 
plt.title(f'Patient 01: Raw PPG Signal ({ppg_col})')
plt.ylabel('Amplitude')
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', linewidth=0.5)

# Plot Respiration
plt.subplot(2, 1, 2)
plt.plot(df[resp_col][:500], color='blue', label='True Respiration')
plt.title(f'Reference Respiration Signal ({resp_col})')
plt.xlabel('Time (Samples)')
plt.ylabel('Chest Extension')
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

print("✅ Success! Environment and Data are ready.")