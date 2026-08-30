# Model Card — Intelligent Transformer Health Monitor (DGA)

## Model Details
- **Type:** XGBoost classifier (n_estimators=200, max_depth=4)
- **Calibration:** Temperature scaling, T* = 1.7334
- **Abstention threshold:** 0.50 confidence
- **Features:** 5 raw gases + 4 engineered ratios (9 total)

## Intended Use
Automated fault triage for oil-filled power transformers via dissolved gas analysis.
Outputs one of 7 IEC TC10 fault classes or routes to engineer review.

## Training Data
- Source: DGA_train.xlsx — 582 rows after deduplication (2 removed)
- Classes: Normal, PD, D1, D2, T1, T2, T3 (balanced, 10 per class in test set)

## Evaluation (70-row sealed test set)
| Metric            | Value  |
|-------------------|--------|
| Overall accuracy  | 80.0 % |
| Macro F1          | 0.80   |
| ECE (before cal.) | 0.158  |
| ECE (after cal.)  | 0.106  |
| Coverage @ 100 %  | 94.3 % |
| Abstention rate   | 5.7 %  |

## Per-Class F1
| Class  | F1   |
|--------|------|
| Normal | 0.75 |
| PD     | 0.70 |
| D1     | 0.57 |
| D2     | 0.80 |
| T1     | 0.91 |
| T2     | 0.95 |
| T3     | 0.91 |

## Limitations
- D1 (low-energy discharge) is the weakest class (F1=0.57); treat predictions with caution.
- IEC TC10 benchmark: 42/49 rows overlap with training. Honest hold-out has only 6 rows.
- Model trained on lab dataset; field performance may differ.

## Ethical Considerations
Model outputs are decision-support only. All abstained samples and D1 predictions
must be reviewed by a qualified engineer before maintenance action.
