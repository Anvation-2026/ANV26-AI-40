# MedGuard AI — Model Validation Report

**Schema Version:** `1.0`  
**Model Version:** `resnet18-pneumoniamnist224-v1+9bda2caa`  
**Generated At:** `2026-10-08T06:05:34.372489+00:00`  
**Hardware Device:** `cuda`  
**Dataset:** PneumoniaMNIST+ (224x224) (224x224)

---

## 1. Executive Summary & Test Performance

Primary evaluation on official **Test split** (N = 624):

| Metric | Test Value (Calibrated) | 95% Bootstrap CI | Validation Value |
|---|---|---|---|
| **AUROC** | **0.9929** | [0.9880, 0.9968] | 0.9990 |
| **Accuracy** | **0.9551** | [0.9375, 0.9712] | 0.9790 |
| **Sensitivity (Recall)** | **0.9897** | [0.9791, 1.0000] | 0.9769 |
| **Specificity** | **0.8974** | [0.8583, 0.9342] | 0.9852 |
| **Precision** | **0.9415** | - | 0.9948 |
| **F1 Score** | **0.9650** | - | 0.9857 |
| **False-Negative Rate** | **0.0103** | - | 0.0231 |

### Confusion Matrix (Test Split)
```
                Predicted Normal    Predicted Pneumonia
Actual Normal        210                 24
Actual Pneumonia     4                   386
```

---

## 2. Calibration & Temperature Scaling

Fitted on **Validation Split** (N = 524):
- **Learned Temperature $T$:** `1.5773`
- **Validation ECE:** `0.0134` $\to$ `0.0118`
- **Test ECE:** `0.0349` $\to$ `0.0278`
- **Test Brier Score:** `0.0351` | **Test NLL:** `0.1325`

---

## 3. Selective Prediction & Uncertainty Abstention

Threshold $\tau_{\text{accept}}$ fitted on validation set: **0.5618**

- **Coverage on Test Set:** **98.6%**
- **Abstained (Uncertain):** **9** samples
- **Accuracy on Accepted Samples:** **0.9577**
- **False Negatives Caught by Abstention:** 0 out of 4 total FNs

---

## 4. Limitations & Disclaimers

1. Educational research prototype; not a medical device and not a clinical diagnosis.
2. Trained and evaluated on PneumoniaMNIST (pediatric chest X-rays, downsampled public benchmark); performance on adults, other hospitals, scanners, or populations is unknown.
3. Calibrated probabilities reflect average reliability on validation data, not a guarantee that any individual prediction is correct.
4. Grad-CAM heatmaps show where the model's output was sensitive; they do not confirm a medical finding.
5. Image-quality and out-of-distribution checks are heuristic; some unsuitable or unsupported images may pass, and some valid images may be rejected.
6. The system cannot guarantee zero false negatives.
7. No demographic metadata is available, so subgroup fairness could not be evaluated.
8. Validation and test splits may differ in distribution; reported test performance is the unbiased estimate.
9. Quality-degradation tests use synthetic degradations created for this project.
