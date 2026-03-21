import asyncio
import csv
import json
import os
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# --- Signal processing ---
from scipy.signal import butter, filtfilt

# --- Model inference ---
import torch
from model import AttentionBiLSTM, ANOMALY_CLASSES

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# Constants (must match dataset.py exactly)
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 125        # Hz
SEGMENT_LEN  = 125 * 30   # 3750 samples = 30 seconds
FRAME_POINTS = 50         # data points sent per WebSocket frame
DEVICE       = torch.device("cpu")

# ─────────────────────────────────────────────────────────────
# Load trained model once at startup
# Falls back to synthetic if .pth is missing
# ─────────────────────────────────────────────────────────────
_model = None
_model_loaded = False

def _try_load_model():
    global _model, _model_loaded
    path = "centralized_model.pth"
    if not os.path.exists(path):
        print("[XAI] No model file found — using synthetic waveform fallback.")
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

# ─────────────────────────────────────────────────────────────
# Rolling PPG signal buffer
# ─────────────────────────────────────────────────────────────
_signal_buffer = deque(maxlen=SEGMENT_LEN * 4)

def _bandpass(data):
    nyq = 0.5 * SAMPLE_RATE
    b, a = butter(4, [0.1 / nyq, 0.5 / nyq], btype="band")
    return filtfilt(b, a, data)

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
        raw   = df[col].values.astype(np.float32)
        clean = _bandpass(raw)
        clean = (clean - clean.mean()) / (clean.std() + 1e-6)
        _signal_buffer.extend(clean.tolist())
        print(f"[XAI] Loaded real PPG from {files[0]} ({len(clean)} samples).")
        return True
    except Exception as e:
        print(f"[XAI] Could not load real data: {e}")
        return False

def _fill_synthetic(n=SEGMENT_LEN * 2):
    t = np.linspace(0, n / SAMPLE_RATE, n)
    sig = (
        np.sin(2 * np.pi * 0.3 * t)
        + 0.4 * np.sin(2 * np.pi * 0.6 * t)
        + 0.15 * np.sin(2 * np.pi * 1.2 * t + 0.5)
        + 0.05 * np.random.randn(n)
    ).astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    _signal_buffer.extend(sig.tolist())
    print("[XAI] Synthetic PPG buffer initialised.")

# Prime the buffer on startup
if not _try_fill_from_file():
    _fill_synthetic()

# ─────────────────────────────────────────────────────────────
# Real-time inference
# ─────────────────────────────────────────────────────────────
def _run_inference():
    """
    Returns (wave_snippet, attention_snippet, anomaly_result).
    anomaly_result: dict with class_id, name, confidence, severity, color, rr_pred.
    Returns None for anomaly_result when model is not loaded (synthetic fallback).
    """
    buf = list(_signal_buffer)
    wave_snippet = [round(float(v), 4) for v in buf[-FRAME_POINTS:]]

    if not _model_loaded or len(buf) < SEGMENT_LEN:
        attn = [round(1.0 if v > 0.6 else 0.05, 3) for v in wave_snippet]
        return wave_snippet, attn, None

    segment = np.array(buf[-SEGMENT_LEN:], dtype=np.float32)
    x = torch.tensor(segment).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rr_pred, alpha, anomaly_logits = _model(
            x, return_attention=True, return_anomaly=True
        )

    # Coarsen attention 3750 -> FRAME_POINTS
    alpha_np = alpha.squeeze(0).cpu().numpy()
    block    = SEGMENT_LEN // FRAME_POINTS
    coarse   = [float(alpha_np[i*block:(i+1)*block].mean()) for i in range(FRAME_POINTS)]
    lo, hi   = min(coarse), max(coarse)
    span     = (hi - lo) if (hi - lo) > 1e-8 else 1.0
    attn     = [round((v - lo) / span, 4) for v in coarse]

    # Anomaly classification
    probs      = torch.softmax(anomaly_logits, dim=-1).squeeze(0).cpu().numpy()
    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])
    rr_val     = float(rr_pred.squeeze())

    anomaly_result = {
        "class_id":   class_id,
        "name":       ANOMALY_CLASSES[class_id]["name"],
        "confidence": round(confidence, 4),
        "severity":   ANOMALY_CLASSES[class_id]["severity"],
        "color":      ANOMALY_CLASSES[class_id]["color"],
        "rr_pred":    round(rr_val, 2),
        "all_probs":  [round(float(p), 4) for p in probs],
    }
    return wave_snippet, attn, anomaly_result

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def get_dashboard():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h2>Error: dashboard.html not found!</h2>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Streams at 2 fps:
      - FL simulation metrics  (from simulation_log.csv)
      - Real PPG waveform      (from BIDMC signal buffer)
      - Real attention weights (from AttentionBiLSTM inference)
    Falls back gracefully if model or data files are missing.
    """
    await websocket.accept()

    fp32_mb    = 5.4
    int8_mb    = 1.2
    epsilon    = -1.0   # -1 = DP not yet reported
    delta      = 1e-5
    dp_enabled = False

    try:
        while True:
            # ── 1. FL metrics from log ───────────────────────────────
            current_round, mae, c0_rqi, c1_rqi = 0, 0.0, 0.0, 0.0
            anomaly_counts = [0, 0, 0, 0, 0]   # Normal, Brady, Apnea, Tachy, SevTachy

            if os.path.exists("simulation_log.csv"):
                try:
                    with open("simulation_log.csv", "r") as f:
                        lines = list(csv.reader(f))
                    if len(lines) > 1:
                        row           = lines[-1]
                        # Column order: Round(0) MAE(1) RMSE(2) C0_RQI(3) C1_RQI(4)
                        #               FP32(5) INT8(6) Epsilon(7) Delta(8) DP(9)
                        #               Ano_Normal(10) Ano_Brady(11) Ano_Apnea(12)
                        #               Ano_Tachy(13) Ano_SevTachy(14)
                        current_round = int(row[0])
                        mae           = float(row[1])
                        c0_rqi        = float(row[3]) if len(row) > 3 else c0_rqi
                        c1_rqi        = float(row[4]) if len(row) > 4 else c1_rqi
                        fp32_mb       = float(row[5]) if len(row) > 5 else fp32_mb
                        int8_mb       = float(row[6]) if len(row) > 6 else int8_mb
                        epsilon       = float(row[7]) if len(row) > 7 else epsilon
                        delta         = float(row[8]) if len(row) > 8 else delta
                        dp_enabled    = bool(int(float(row[9]))) if len(row) > 9 else dp_enabled
                        anomaly_counts = [
                            int(float(row[10])) if len(row) > 10 else 0,
                            int(float(row[11])) if len(row) > 11 else 0,
                            int(float(row[12])) if len(row) > 12 else 0,
                            int(float(row[13])) if len(row) > 13 else 0,
                            int(float(row[14])) if len(row) > 14 else 0,
                        ]
                except Exception:
                    pass  # CSV mid-write — skip this frame

            # ── 2. Replenish buffer if low ───────────────────────────
            if len(_signal_buffer) < FRAME_POINTS * 4:
                if not _try_fill_from_file():
                    _fill_synthetic()

            # ── 3. Run model inference on real signal ────────────────
            wave_data, attention_data, anomaly_result = _run_inference()

            # Advance sliding window: consume oldest points
            consume = min(FRAME_POINTS // 2, len(_signal_buffer) - SEGMENT_LEN)
            for _ in range(max(0, consume)):
                _signal_buffer.popleft()

            # ── 4. Broadcast ─────────────────────────────────────────
            payload = {
                "round":          current_round,
                "mae":            mae,
                "c0_rqi":         c0_rqi,
                "c1_rqi":         c1_rqi,
                "wave":           wave_data,
                "attention":      attention_data,
                "fp32_mb":        fp32_mb,
                "int8_mb":        int8_mb,
                "xai_live":       _model_loaded,
                "epsilon":        epsilon,
                "delta":          delta,
                "dp_enabled":     dp_enabled,
                "anomaly":        anomaly_result,   # live inference dict or None
                "anomaly_counts": anomaly_counts,   # per-round FL distribution [N,B,A,T,S]
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"[WS] Client disconnected: {e}")