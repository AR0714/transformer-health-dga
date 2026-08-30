# Intelligent Transformer Health Monitoring Using DGA

Final-year B.Tech (EE) project — KIIT University

## What This Does
Classifies power-transformer faults from dissolved gas analysis (DGA) into
7 IEC TC10 fault classes using a calibrated XGBoost model with an abstention layer.

## Key Results
| Step | Result |
|------|--------|
| Model accuracy (sealed 70-row test) | 80.0 % |
| Calibration ECE (before -> after)   | 0.158 -> 0.106 |
| Coverage at 100 % precision         | ~55 % |
| Abstention rate                     | 5.7 % (4/70 routed to engineer) |
| Strongest class (T2)                | F1 = 0.95 |
| Weakest class (D1)                  | F1 = 0.57 |
| Data leakage finding                | 42/49 IEC TC10 rows overlap train |

## Repository Structure

    transformer-health-dga/
    +-- data/       DGA_train.xlsx, DGA_test_unseen.xlsx, IEC_TC_10_data.xlsx
    +-- models/     xgb_dga_base.joblib, xgb_dga_calibrated.joblib
    +-- notebooks/  01_load_and_look.ipynb (all steps)
    +-- reports/    confusion_matrix.png, shap_summary.png,
    |               shap_per_class.png, risk_coverage_curve.png, MODEL_CARD.md
    +-- src/        calibration.py (CalibratedDGAModel + abstention)

## How to Run

    conda activate base
    cd notebooks
    jupyter notebook 01_load_and_look.ipynb

Run all cells top to bottom.

## Fault Classes
Normal | PD | D1 | D2 | T1 | T2 | T3

## Author
Ankit Raj — KIIT University, 2026