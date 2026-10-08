# MedGuard AI — Model Validation Report

**Schema Version:** `1.0`  
**Model Version:** `resnet18-pneumoniamnist224-v1+2c4f0760`  
**Generated At:** `2026-10-08T12:00:44.740392+00:00`  
**Hardware Device:** `cpu`  
**Dataset:** PneumoniaMNIST+ (224x224) (224x224)

---

## 1. Executive Summary & Test Performance

Primary evaluation on official **Test split** (N = 624):

| Metric | Test Value (Calibrated) | 95% Bootstrap CI | Validation Value |
|---|---|---|---|
| **AUROC** | **0.9816** | [0.9711, 0.9910] | 0.9982 |
| **Accuracy** | **0.9087** | [0.8814, 0.9311] | 0.9752 |
| **Sensitivity (Recall)** | **0.9974** | [0.9918, 1.0000] | 0.9846 |
| **Specificity** | **0.7607** | [0.7000, 0.8136] | 0.9481 |
| **Precision** | **0.8742** | - | 0.9821 |
| **F1 Score** | **0.9317** | - | 0.9833 |
| **False-Negative Rate** | **0.0026** | - | 0.0154 |

### Confusion Matrix (Test Split)
```
                Predicted Normal    Predicted Pneumonia
Actual Normal        178                 56
Actual Pneumonia     1                   389
```

---

## 2. Calibration & Temperature Scaling

Fitted on **Validation Split** (N = 524):
- **Learned Temperature $T$:** `2.1718`
- **Validation ECE:** `0.0172` $\to$ `0.0081`
- **Test ECE:** `0.0756` $\to$ `0.0570`
- **Test Brier Score:** `0.0744` | **Test NLL:** `0.3303`

---

## 3. Selective Prediction & Uncertainty Abstention

Threshold $\tau_{\text{accept}}$ fitted on validation set: **0.5063**

- **Coverage on Test Set:** **100.0%**
- **Abstained (Uncertain):** **0** samples
- **Accuracy on Accepted Samples:** **0.9087**
- **False Negatives Caught by Abstention:** 0 out of 1 total FNs

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
