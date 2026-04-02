# Fed-Breath: Evaluation Results

Dual-benchmark results for the Fed-Breath Respiratory Rate Estimation system.

**Model**: AttentionBiLSTM (BiLSTM + Attention + MC Dropout, dual-head)
**Training**: Multi-task loss = MSE(RR) + 0.3 * CE(anomaly)
**Preprocessing**: 125 Hz, bandpass 0.1-0.5 Hz, 30s segments (3750 samples), z-score

---

## 1. BIDMC Dataset (5-Fold Cross-Validation, N=53)

| Metric         | Mean    | +/- Std | Unit  |
|----------------|---------|---------|-------|
| MAE            | 2.32    | 0.27    | BrPM  |
| RMSE           | 3.13    | 0.63    | BrPM  |
| MBE (Bias)     | -0.69   | 1.15    | BrPM  |
| Within +/-2    | 62.31   | 9.75    | %     |
| Within +/-5    | 88.46   | 4.32    | %     |
| BA Bias        | -0.69   | 1.15    | BrPM  |
| BA SD          | 2.89    | 0.95    | BrPM  |
| Anomaly Acc    | 82.92   | 3.96    | %     |

**Paper-ready format:**
- MAE  = 2.32 +/- 0.27 BrPM
- RMSE = 3.13 +/- 0.63 BrPM
- Bias = -0.69 BrPM, 95% LoA = [-6.35, 4.97] BrPM

---

## 2. Single-Split Evaluation (BIDMC, N=10 val)

| Metric              | Value   | Unit  |
|---------------------|---------|-------|
| MAE                 | 1.92    | BrPM  |
| RMSE                | 2.38    | BrPM  |
| R2                  | -0.03   |       |
| MBE                 | 0.42    | BrPM  |
| Within +/-2         | 70.0    | %     |
| Within +/-5         | 90.0    | %     |

### Bland-Altman Agreement
| Metric              | Value   | Unit  |
|---------------------|---------|-------|
| Bias                | 0.42    | BrPM  |
| SD                  | 2.47    | BrPM  |
| 95% LoA Upper       | 5.25    | BrPM  |
| 95% LoA Lower       | -4.41   | BrPM  |
| Within LoA          | 90.0    | %     |
| Clinical test       | REVIEW  | (LoA width = 9.66 > 6 BrPM) |

### MC Dropout Uncertainty (30 samples, mc_dropout_p=0.3)
| Metric              | Value   | Unit  |
|---------------------|---------|-------|
| 95% CI Coverage     | 90.0    | %     |
| Mean CI Width       | 5.88    | BrPM  |
| Mean Sigma          | 1.50    | BrPM  |

> **Calibration improvement**: Coverage increased from 30% (mc_dropout_p=0.1)
> to 90% (mc_dropout_p=0.3 + temperature scaling). Target is 95%.

### Anomaly Detection
| Class              | Precision | Recall | F1    | Support |
|--------------------|-----------|--------|-------|---------|
| Normal             | 0.900     | 1.000  | 0.947 | 9       |
| Bradypnea          | 0.000     | 0.000  | 0.000 | 1       |
| Apnea              | 0.000     | 0.000  | 0.000 | 0       |
| Tachypnea          | 0.000     | 0.000  | 0.000 | 0       |
| Severe Tachypnea   | 0.000     | 0.000  | 0.000 | 0       |

Overall accuracy: 90.0%

> Note: BIDMC dataset is predominantly Normal-class (RR 12-20 BrPM).
> CapnoBase integration will add pathological cases (RR <10 and >40 BrPM).

---

## 3. CapnoBase Dataset

> **Status: PENDING** -- CapnoBase .mat files need to be placed in `data/capnobase/`.

Once data is available, run:
```bash
# Cross-dataset: train BIDMC, test CapnoBase
python train_multidataset.py --datasets bidmc --eval-on capnobase --epochs 20

# Joint training: BIDMC + CapnoBase
python train_multidataset.py --datasets bidmc capnobase --eval-on capnobase --epochs 25

# K-fold on CapnoBase
python evaluate.py --model model_bidmc_capnobase.pth --dataset capnobase --kfold --mc-samples 20
```

Expected improvements with CapnoBase:
- Wider RR range coverage (<10 to >40 BrPM) for anomaly detection
- CO2-derived reference (more accurate than BIDMC impedance pneumography)
- Better non-Normal class metrics (Bradypnea, Apnea, Tachypnea)

---

## 4. Strategy Comparison (FedAvg vs FedProx vs FedRQI)

From existing FL simulation (20 rounds, 2 clients, BIDMC):

| Strategy | Final MAE (BrPM) | Final RMSE (BrPM) |
|----------|------------------|--------------------|
| FedAvg   | ~14.24           | ~14.50             |
| FedProx  | ~9.99            | ~10.16             |
| FedRQI   | ~10.89           | ~11.22             |

> FedProx with mu=0.01 shows fastest convergence on BIDMC.
> Strategy comparison with CapnoBase data (more heterogeneous) is expected
> to show larger FedProx advantage due to greater non-IID distribution.

To run mu hyperparameter sweep:
```bash
python fedprox_client.py --compare --mu-sweep
```

---

## 5. Architecture Summary

| Component         | Detail                                    |
|-------------------|-------------------------------------------|
| Encoder           | 2-layer BiLSTM (hidden=64)                |
| Attention         | Soft attention (linear + softmax)         |
| RR Head           | Linear(128, 1) -- regression              |
| Anomaly Head      | Linear(128, 5) -- 5-class classification  |
| MC Dropout        | p=0.3, applied after attention pooling    |
| Temperature       | Post-hoc calibration parameter            |
| Preprocessing     | 125 Hz, BP 0.1-0.5 Hz, 30s, z-score      |
| Multi-task Loss   | MSE + 0.3 * CrossEntropy                  |
| Edge Deployment   | INT8 quantization via torch.quantization  |
| Privacy           | DP-SGD via Opacus (sigma=1.0, C=1.0)     |
| FL Strategies     | FedAvg, FedProx (mu sweep), FedRQI+BFT   |

---

## 6. New Features Implemented

1. **MC Dropout Calibration (Priority 1)**: mc_dropout_p increased from 0.1 to 0.3,
   achieving 90% CI coverage (up from 30%)
2. **Temperature Scaling (Priority 4)**: Post-hoc calibration parameter for
   uncertainty intervals without retraining
3. **Personalized FL (Priority 3)**: `client.py` `personalize()` method freezes
   BiLSTM/attention and fine-tunes only fc_rr + fc_anomaly heads
4. **FedProx mu Sweep (Priority 2)**: `--mu-sweep` flag tests mu in [0.001, 0.01, 0.1]
5. **Multi-dataset Training**: New `train_multidataset.py` for BIDMC/CapnoBase
   joint training and cross-dataset evaluation
6. **CapnoBase Loader**: Auto-discovery of PPG signals and RR labels from .mat files
   with 300->125 Hz resampling
