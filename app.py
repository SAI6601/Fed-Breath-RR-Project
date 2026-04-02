import asyncio
import csv
import json
import os
import math
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# --- Signal processing ---
from scipy.signal import butter, sosfiltfilt
import torch
from model import AttentionBiLSTM, ANOMALY_CLASSES

app = FastAPI()

SAMPLE_RATE  = 125        
SEGMENT_LEN  = 125 * 30   
FRAME_POINTS = 50         
DEVICE       = torch.device("cpu")

_model = None
_model_loaded = False

def _try_load_model():
    global _model, _model_loaded
    path = "model_bidmc_capnobase.pth"
    if not os.path.exists(path):
        print(f"[XAI] No model file found at {path} — using synthetic waveform fallback.")
        return
    try:
        m = AttentionBiLSTM().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        _model = m
        _model_loaded = True
        print("[XAI] AttentionBiLSTM loaded — real inference active.")
    except Exception as e:
        print(f"[XAI] Model load failed ({e}) — using synthetic fallback.")

_try_load_model()


_signal_buffer = deque(maxlen=SEGMENT_LEN * 4)
_visual_buffer = deque(maxlen=SEGMENT_LEN * 4)
_read_cursor = 0  

def _bandpass(data):
    nyq = 0.5 * SAMPLE_RATE
    sos = butter(4, [0.1 / nyq, 0.5 / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data)

def _try_fill_from_file():
    raw_dir = os.path.join("data", "raw")
    if not os.path.isdir(raw_dir):
        return False
    try:
        import pandas as pd
        files = [f for f in os.listdir(raw_dir) if f.endswith("Signals.csv")]
        if not files:
            return False
        df    = pd.read_csv(os.path.join(raw_dir, files[0]))
        col   = [c for c in df.columns if "PLETH" in c.upper()][0]
        
        raw   = df[col].values.astype(np.float64) 
        raw   = np.nan_to_num(raw, nan=0.0)
        
        # Model buffer
        _signal_buffer.extend(raw.tolist())
        
        # Visual buffer (APG ECG-like transform)
        d1 = np.gradient(raw)
        apg = -np.gradient(d1)
        std_apg = np.nanstd(apg)
        vis = (apg - np.nanmean(apg)) / (std_apg if std_apg > 1e-6 else 1.0)
        vis = np.nan_to_num(vis, nan=0.0)
        _visual_buffer.extend(vis.tolist())
        print(f"[XAI] Loaded real PPG from {files[0]} ({len(vis)} samples).")
        return True
    except Exception as e:
        print(f"[XAI] Could not load real data: {e}")
        return False

def _fill_synthetic(n=SEGMENT_LEN * 8):
    """
    Realistic ECG-style PPG waveform:
    - 72 BPM cardiac pulses (P-Q-R-S-T morphology, sharp R spike)
    - 15 BrPM breathing baseline modulation (0.25 Hz)
    """
    t      = np.linspace(0, n / SAMPLE_RATE, n)
    hr_hz  = 1.2    # 72 BPM
    br_hz  = 0.25   # 15 BrPM

    breath_env    = 0.25 * np.sin(2 * np.pi * br_hz * t)
    cardiac       = np.zeros(n, dtype=np.float64)
    beat_period   = int(SAMPLE_RATE / hr_hz)   # ~104 samples

    for beat_start in range(0, n, beat_period):
        x = np.arange(n) - beat_start
        p  = int(0.15  * SAMPLE_RATE)
        q  = int(0.02  * SAMPLE_RATE)
        s  = int(0.025 * SAMPLE_RATE)
        tw = int(0.30  * SAMPLE_RATE)
        w_r  = (0.008 * SAMPLE_RATE)**2
        w_p  = (0.025 * SAMPLE_RATE)**2
        w_q  = (0.012 * SAMPLE_RATE)**2
        w_s  = (0.015 * SAMPLE_RATE)**2
        w_t  = (0.050 * SAMPLE_RATE)**2
        cardiac += 0.10 * np.exp(-0.5 * (x + p)**2 / w_p)   # P wave
        cardiac -= 0.07 * np.exp(-0.5 * (x + q)**2 / w_q)   # Q dip
        cardiac += 1.20 * np.exp(-0.5 * x**2          / w_r) # R peak (spike)
        cardiac -= 0.18 * np.exp(-0.5 * (x - s)**2    / w_s) # S dip
        cardiac += 0.28 * np.exp(-0.5 * (x - tw)**2   / w_t) # T wave

    sig  = (cardiac + breath_env).astype(np.float32)
    sig += 0.015 * np.random.randn(n).astype(np.float32)
    sig  = np.clip(sig, -2.5, 2.5)
    sig  = sig / (np.abs(sig).max() + 1e-6) * 1.35
    _signal_buffer.extend(sig.tolist())
    print("[XAI] ECG-style synthetic PPG buffer initialised.")

if not _try_fill_from_file():
    _fill_synthetic()

# FIX 2: A truly bulletproof float converter
def safe_float(val, precision=4):
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return round(f, precision)
    except (ValueError, TypeError):
        return 0.0

def _run_inference():
    buf = list(_signal_buffer)
    global _read_cursor
    # Send CONSECUTIVE raw samples — guaranteed smooth continuity at client
    ADVANCE      = FRAME_POINTS   
    buf_list     = list(buf)
    vis_list     = list(_visual_buffer)
    buf_len      = len(buf_list)
    _read_cursor = _read_cursor % max(1, buf_len - FRAME_POINTS)
    wave_snippet = [safe_float(v) for v in vis_list[_read_cursor:_read_cursor + FRAME_POINTS]]
    
    end_idx = _read_cursor + FRAME_POINTS
    start_idx = end_idx - SEGMENT_LEN
    _read_cursor = (_read_cursor + ADVANCE) % max(1, buf_len - FRAME_POINTS)

    if not _model_loaded or len(buf) < min(SEGMENT_LEN, FRAME_POINTS * 4):
        if wave_snippet:
            peak_val = max(abs(v) for v in wave_snippet)
            thresh   = peak_val * 0.72    # mark R-peak spikes (top ~10%)
            attn     = [safe_float(1.0 if abs(v) >= thresh else 0.03) for v in wave_snippet]
        else:
            attn = [0.03] * FRAME_POINTS
        # Synthetic fallback anomaly result so the monitor/bars always render
        synthetic_anomaly = {
            "class_id": 0,
            "name": ANOMALY_CLASSES[0]["name"],
            "confidence": 0.0,
            "severity": ANOMALY_CLASSES[0]["severity"],
            "color": ANOMALY_CLASSES[0]["color"],
            "rr_pred": 0.0,
            "all_probs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        return wave_snippet, attn, synthetic_anomaly

    if start_idx < 0:
        segment = np.concatenate((np.zeros(-start_idx), buf_list[0:end_idx]))
    else:
        segment = np.array(buf_list[start_idx:end_idx], dtype=np.float32)
        
    segment = np.nan_to_num(segment, nan=0.0)
    
    # Neural net needs the 0.1-0.5Hz bandpass!
    inf_segment = _bandpass(segment)
    inf_std = np.std(inf_segment)
    inf_segment = (inf_segment - np.mean(inf_segment)) / (inf_std if inf_std > 1e-6 else 1.0)
    
    x = torch.tensor(inf_segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rr_pred, alpha, anomaly_logits = _model(
            x, return_attention=True, return_anomaly=True
        )

    alpha_np = alpha.squeeze(0).cpu().numpy()
    
    # Get attention precisely aligned with our 50-point raw wave snippet
    snippet_alpha = alpha_np[-FRAME_POINTS:]
    window_max = alpha_np.max()
    window_min = alpha_np.min()
    span = (window_max - window_min) if (window_max - window_min) > 1e-8 else 1.0
    attn = [safe_float((v - window_min) / span) for v in snippet_alpha]

    probs      = torch.softmax(anomaly_logits, dim=-1).squeeze(0).cpu().numpy()
    probs      = np.nan_to_num(probs, nan=0.0)
    class_id   = int(np.argmax(probs))
    confidence = safe_float(probs[class_id])
    rr_val     = safe_float(rr_pred.squeeze(), precision=2)

    anomaly_result = {
        "class_id":   class_id,
        "name":       ANOMALY_CLASSES[class_id]["name"],
        "confidence": confidence,
        "severity":   ANOMALY_CLASSES[class_id]["severity"],
        "color":      ANOMALY_CLASSES[class_id]["color"],
        "rr_pred":    rr_val,
        "all_probs":  [safe_float(p) for p in probs],
    }
    return wave_snippet, attn, anomaly_result

@app.get("/")
async def get_dashboard():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h2>Error: dashboard.html not found!</h2>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    fp32_mb    = 5.4
    int8_mb    = 1.2
    epsilon    = -1.0
    delta      = 1e-5
    dp_enabled = False
    
    last_sent_round = -1

    try:
        while True:
            # --- History Burst ---
            if os.path.exists("simulation_log.csv"):
                try:
                    with open("simulation_log.csv", "r") as f:
                        lines = list(csv.reader(f))
                    if len(lines) > 1:
                        for row in lines[1:]: # Skip header
                            if len(row) < 2: continue # Skip malformed rows
                            
                            current_round = int(safe_float(row[0]))
                            if current_round > last_sent_round:
                                mae           = safe_float(row[1])
                                # CSV: Round,MAE,RMSE,C0_RQI,C1_RQI,FP32_MB,INT8_MB,Epsilon,Delta,DP_Enabled,Ano_0..4
                                #       0     1   2     3      4       5       6       7       8      9        10..14
                                c0_rqi        = safe_float(row[3]) if len(row) > 3 else 0.0
                                c1_rqi        = safe_float(row[4]) if len(row) > 4 else 0.0
                                fp32_val      = safe_float(row[5]) if len(row) > 5 else fp32_mb
                                int8_val      = safe_float(row[6]) if len(row) > 6 else int8_mb
                                eps_val       = safe_float(row[7]) if len(row) > 7 else epsilon
                                delta_val     = safe_float(row[8]) if len(row) > 8 else delta
                                dp_val        = int(safe_float(row[9])) if len(row) > 9 else 0
                                dp_enabled    = bool(dp_val)
                                
                                hist_payload = {
                                    "round": current_round, "mae": mae, "c0_rqi": c0_rqi, "c1_rqi": c1_rqi,
                                    "fp32_mb": fp32_val, "int8_mb": int8_val, "epsilon": eps_val, "delta": delta_val,
                                    "dp_enabled": dp_enabled, "anomaly_counts": [int(safe_float(row[10])) if len(row)>10 else 0, int(safe_float(row[11])) if len(row)>11 else 0, int(safe_float(row[12])) if len(row)>12 else 0, int(safe_float(row[13])) if len(row)>13 else 0, int(safe_float(row[14])) if len(row)>14 else 0],
                                    "wave": None, "attention": None 
                                }
                                await websocket.send_text(json.dumps(hist_payload))
                                last_sent_round = current_round
                                await asyncio.sleep(0.05) 
                except Exception:
                    pass

            # --- Live Stream ---
            if len(_signal_buffer) < SEGMENT_LEN:
                if not _try_fill_from_file():
                    _fill_synthetic()

            try:
                wave_snippet, attn, anomaly_result = await asyncio.to_thread(_run_inference)
            except Exception as e:
                print(f"[WS] Inference error ignored: {e}")
                wave_snippet, attn, anomaly_result = [], [0.03]*FRAME_POINTS, {}

            # Don't drain buffer - sliding window keeps signal healthy
            # Only trim to keep buffer from growing unbounded
            if len(_signal_buffer) > SEGMENT_LEN * 3:
                excess = len(_signal_buffer) - SEGMENT_LEN * 2
                for _ in range(excess):
                    _signal_buffer.popleft()

            live_payload = {
                "round":          last_sent_round if last_sent_round > 0 else 0,
                "wave":           wave_snippet,
                "attention":      attn,
                "xai_live":       _model_loaded,
                "anomaly":        anomaly_result,
            }

            await websocket.send_text(json.dumps(live_payload))
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"[WS] Client disconnected: {e}")