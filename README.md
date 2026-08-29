# Intelligent Transformer Health Monitoring via Dissolved Gas Analysis (DGA)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Standards](https://img.shields.io/badge/Standards-IEC%2060599%20%7C%20IEEE%20C57.104-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Part%201%20Complete%20%7C%20Part%202%20In%20Progress-orange.svg)]()

> **A complete, production-oriented pipeline for diagnosing internal faults in power transformers using dissolved gas analysis — combining classical IEC/IEEE diagnostic methods with calibrated machine learning and explainable AI.**

---

## Table of Contents

1. [Why This Matters](#1-why-this-matters)
2. [What is DGA?](#2-what-is-dga)
3. [The 7 Fault Classes](#3-the-7-fault-classes)
4. [System Architecture](#4-system-architecture)
5. [What We Built — Part 1](#5-what-we-built--part-1)
6. [Results](#6-results)
7. [Key Findings](#7-key-findings)
8. [Evaluation Honesty](#8-evaluation-honesty)
9. [Repository Structure](#9-repository-structure)
10. [How to Run](#10-how-to-run)
11. [Dataset](#11-dataset)
12. [Standards & References](#12-standards--references)
13. [Roadmap — Part 2](#13-roadmap--part-2)
14. [About](#14-about)

---

## 1. Why This Matters

A large power transformer is one of the most critical — and expensive — assets in an electrical grid.

- **Cost:** USD 2 million to 10 million per unit
- **Lead time:** 18 to 24 months to manufacture and replace
- **Consequence of failure:** Widespread blackouts, grid instability, risk to human life, and catastrophic financial loss
- **Global scale:** There are over 3 million distribution and power transformers operating worldwide

Despite this, most transformers today are maintained on fixed schedules — not based on their actual health condition. This is called *time-based maintenance*, and it leads to two problems: either the transformer fails unexpectedly before maintenance is due, or money is wasted servicing equipment that is still in good health.

**This project implements *condition-based monitoring* (CBM)** — the practice of reading real signals from the transformer itself to decide whether it needs attention, and what kind of fault, if any, is developing inside it.

---

## 2. What is DGA?

Dissolved Gas Analysis (DGA) is the most widely used, internationally standardized technique for detecting internal faults in oil-filled power transformers. It is endorsed by IEC 60599 and IEEE C57.104 as the primary on-site diagnostic method.

**How it works in plain language:**

Inside a transformer, high-voltage electricity flows through copper windings immersed in insulating mineral oil. When a fault develops — an electrical arc, excessive heat, partial discharge — it causes the oil and paper insulation to decompose. This decomposition releases specific gases, which dissolve into the oil. By extracting a small oil sample and measuring the concentration of these gases in parts per million (ppm), a trained engineer — or a diagnostic system — can infer what type of fault is occurring, and how severe it is.

Think of it like a blood test for a transformer. Just as different diseases produce different markers in human blood, different fault types produce different gas signatures in transformer oil.

**The five key diagnostic gases measured in this project:**

| Gas | Symbol | Plain Language | Associated Fault |
|-----|--------|---------------|-----------------|
| Hydrogen | H₂ | Produced by nearly all fault types | Partial discharge, arcing |
| Methane | CH₄ | Produced by mild overheating of oil | Low-temperature thermal faults |
| Ethane | C₂H₆ | Produced by moderate oil heating | Moderate thermal faults |
| Ethylene | C₂H₄ | Produced by high-temperature oil decomposition | High-temperature thermal faults |
| Acetylene | C₂H₂ | Exclusively produced by high-energy electrical arcing | Electrical discharge (D2) |

Acetylene (C₂H₂) is the most important single gas. Even a few ppm of acetylene in transformer oil is a red flag requiring immediate engineer attention.

---

## 3. The 7 Fault Classes

This project diagnoses seven distinct operational states, following IEC 60599 classification:

| Class | Full Name | What It Means |
|-------|-----------|---------------|
| **Normal** | No fault | Transformer operating within safe parameters |
| **PD** | Partial Discharge | Low-energy electrical discharges in oil or paper — early warning |
| **D1** | Low-Energy Discharge | Sparking without sustained arc — moderate electrical stress |
| **D2** | High-Energy Discharge | Full electrical arcing — severe, urgent attention required |
| **T1** | Thermal Fault < 300°C | Mild overheating — often due to circulating currents |
| **T2** | Thermal Fault 300–700°C | Moderate overheating — paper insulation at risk |
| **T3** | Thermal Fault > 700°C | Severe overheating — oil carbonization, imminent failure risk |

---

## 4. System Architecture

```
                    ┌─────────────────────────────────────┐
                    │        OIL SAMPLE (5 Gases)         │
                    │   H₂, CH₄, C₂H₆, C₂H₄, C₂H₂ (ppm)│
                    └──────────────┬──────────────────────┘
                                   │
               ┌───────────────────┴───────────────────────┐
               │                                           │
     ┌─────────▼──────────┐                   ┌───────────▼────────────┐
     │  CLASSICAL METHODS  │                   │   MACHINE LEARNING     │
     │  (IEC/IEEE Rules)   │                   │   PIPELINE             │
     │                     │                   │                        │
     │  ├─ Key Gas Method  │                   │  ├─ Feature Engineering │
     │  ├─ IEC 60599 Ratios│                   │  │   (9 features)       │
     │  └─ Duval Triangle 1│                   │  ├─ XGBoost Classifier  │
     │                     │                   │  ├─ Isotonic Calibration│
     └─────────┬───────────┘                   └───────────┬────────────┘
               │                                           │
               │      ┌────────────────────────┐           │
               └──────►   DIAGNOSIS RESULT      ◄──────────┘
                      │   Fault Class + Conf.  │
                      └────────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    EXPLAINABILITY (SHAP)     │
                    │  Which gas drove the result? │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    FLEET RISK RANKING        │
                    │  70 transformers sorted by   │
                    │  fault severity + confidence │
                    └─────────────────────────────┘
```

---

## 5. What We Built — Part 1

### 5.1 Classical Diagnostic Methods (Rule-Based Baselines)

Three internationally standardized methods were implemented from first principles in Python, following IEC 60599 and IEEE C57.104 specifications exactly:

**Key Gas Method** (`src/classical/key_gas.py`)
The simplest diagnostic approach: identify which gas is present in the highest concentration and map it to the most likely fault class. Provides a fast first-pass screening before deeper analysis.

**IEC 60599 Three-Ratio Method** (`src/classical/iec_ratios.py`)
Computes three gas ratios (CH₄/H₂, C₂H₂/C₂H₄, C₂H₄/C₂H₆), assigns each a coded value (0, 1, or 2) based on IEC threshold ranges, and maps the resulting three-digit code to a fault class. This is the most widely used classical method in industry.

**Duval Triangle Method 1** (`src/classical/duval.py`)
A graphical technique developed by Michel Duval (IEEE Fellow, CIGRE Working Group Chairman). Converts three normalized gas ratios into a 2D barycentric coordinate on a triangular fault map. The position of the point determines the fault zone. Implemented numerically without any external DGA library.

### 5.2 Machine Learning Pipeline (`notebooks/01_load_and_look.ipynb`)

**Data preparation:** 409-row dataset from IEEE DataPort (DOI: 10.21227/27vy-h479), 7 fault classes, 5 gas features expanded to 9 engineered features (adding log-transformed values and key diagnostic ratios as explicit features).

**Feature engineering:** Beyond the raw 5 gases, the model uses log(H₂+1), log(C₂H₂+1), CH₄/H₂ ratio, C₂H₂/C₂H₄ ratio, and C₂H₄/C₂H₆ ratio — the same ratios used in classical IEC methods — giving the ML model the same chemical intuition that expert engineers use.

**Model training:** XGBoost (eXtreme Gradient Boosting) with stratified 70/30 train-test split, class-balanced weighting to handle unequal fault class frequencies.

**Probability calibration:** Raw XGBoost probabilities were overconfident. Isotonic regression calibration (sklearn's `CalibratedClassifierCV`) was applied to the held-out validation set. Expected Calibration Error (ECE) improved from 0.15 (raw) to 0.10 (calibrated). This is critical for real-world deployment: an overconfident model that says "99% sure it's normal" when it's actually 70% sure can lead to missed fault detections.

**SHAP Explainability:** SHapley Additive exPlanations (SHAP) values were computed for every prediction, identifying which specific gas measurement drove each diagnosis. This is a requirement for operator trust in high-stakes systems — the model cannot function as a black box when a human engineer must decide whether to take a transformer offline.

**Fleet Risk Ranking:** All 70 test-set transformers were scored and ranked by a composite risk metric combining predicted fault severity and calibrated confidence. The output is a CSV (`reports/fleet_risk_ranking.csv`) prioritizing which units need engineer attention first.

---

## 6. Results

Performance evaluated on a **sealed 70-row test set** (10 samples per fault class), held out before any model training or tuning:

| Method | Type | Accuracy on Sealed Test Set | Notes |
|--------|------|-----------------------------|-------|
| Key Gas Method | Classical / Rule-Based | ~47% | Single-gas heuristic; fast but limited |
| IEC 60599 3-Ratio | Classical / Rule-Based | ~55% | Industry-standard method; many samples fall in undefined codes |
| Duval Triangle 1 | Classical / Rule-Based | **57.1%** | Best classical baseline — this is the number ML must beat |
| Random Forest | Machine Learning | 78.6% | Uncalibrated probabilities |
| **XGBoost (Calibrated)** | **Machine Learning** | **80.0%** | **ECE: 0.15 → 0.10 after isotonic calibration** |

**The ML model beats the best classical baseline by +22.9 percentage points** (80.0% vs 57.1%).

The classical baseline of 57.1% is not a failure — it reflects the real-world performance of rule-based DGA diagnosis without any learning from data. Expert engineers routinely achieve 60–70% agreement with each other on the same DGA sample. The ML model surpasses this and, unlike classical methods, provides a calibrated confidence score with every prediction.

---

## 7. Key Findings

**Finding 1 — Acetylene is the dominant diagnostic feature.**
SHAP analysis consistently places C₂H₂ (acetylene) as the #1 most important feature across all fault classes. This is chemically correct: acetylene is exclusively produced by high-energy electrical arcing (D2 faults) and is essentially absent in thermal or normal conditions. The model learned real transformer chemistry from data.

**Finding 2 — Confusion clusters are physically meaningful.**
The confusion matrix does not show random errors. Misclassifications concentrate at the boundaries between discharge fault subtypes: PD↔D1↔D2. This is expected — these fault types exist on a continuum of electrical discharge intensity, and even expert engineers disagree at the boundaries. The model is not confused randomly; it is uncertain exactly where human experts are uncertain.

**Finding 3 — Raw XGBoost is overconfident; calibration matters.**
Before calibration, the model stated probabilities far from the empirically observed frequencies (ECE = 0.15). After isotonic calibration, a stated confidence of 80% corresponds to approximately 80% actual accuracy on held-out samples (ECE = 0.10). In a system that is intended to assist — not replace — human engineers, this calibration is essential.

**Finding 4 — Fleet risk ranking is actionable.**
The top 8 highest-risk units in the 70-transformer test set are all correctly identified as D2 (high-energy electrical discharge) faults. The risk ranking can be delivered to an operations team each morning as a prioritized inspection list — a direct, practical output.

---

## 8. Evaluation Honesty

This section is included deliberately, in the spirit of IEEE and CIGRE technical reporting standards, which require transparent declaration of dataset limitations.

**Test set composition:** The 70-row sealed test set contains exactly 10 samples per fault class (balanced). Real-world transformer fleets are heavily imbalanced — Normal and T3 are common; PD and D1 are rare. The 80% accuracy figure does not represent expected field performance on an unbalanced population. It represents controlled benchmark performance.

**Data leakage — identified and documented:** The IEC TC10 reference dataset (49 rows, a standard DGA benchmark used in academic literature) was found to have 41 out of 49 rows duplicated in the training set (83.7% overlap). This means any performance measured against the full IEC TC10 dataset would be inflated by training-set memorization. This has been identified, documented in `DECISIONS.md`, and will be corrected in Part 2 by constructing a clean, deduplicated holdout.

**Scope:** This is a **diagnosis and triage layer** — it classifies the type of fault from dissolved gas concentrations. It is not a sensor platform, not a real-time monitoring system, and not a replacement for a qualified transformer diagnostic engineer. It is designed to assist engineers by automating the first-pass interpretation of DGA results and prioritizing which units to inspect.

---

## 9. Repository Structure

```
transformer-health-dga/
│
├── notebooks/
│   └── 01_load_and_look.ipynb      # Complete ML pipeline (EDA → Train → SHAP → Fleet Risk)
│
├── src/
│   └── classical/
│       ├── key_gas.py              # Key Gas method (IEC 60599)
│       ├── iec_ratios.py           # Three-Ratio method (IEC 60599 Table 1)
│       └── duval.py                # Duval Triangle Method 1
│
├── reports/
│   ├── calibration_curve.png       # Reliability diagram: raw vs calibrated probabilities
│   ├── confusion_matrix.png        # 7×7 confusion matrix on sealed test set
│   ├── shap_summary.png            # Global SHAP feature importance (bar chart)
│   ├── shap_waterfall.png          # Per-prediction SHAP waterfall (example case)
│   └── fleet_risk_ranking.csv      # 70 transformers ranked by fault risk + confidence
│
├── data/                           # Dataset files (not committed — see Dataset section)
├── models/                         # Saved model artifacts
├── DECISIONS.md                    # Full decision log (all design choices documented)
├── requirements.txt                # Pinned Python dependencies
└── LICENSE                         # MIT License
```

---

## 10. How to Run

### Prerequisites

- Python 3.9 or higher
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/AR0714/transformer-health-dga.git
cd transformer-health-dga

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
# Download from IEEE DataPort: https://ieee-dataport.org/documents/dissolved-gas-analysis-dga-dataset
# DOI: 10.21227/27vy-h479
# Place the CSV file in the data/ folder

# 4. Run the notebook
jupyter notebook notebooks/01_load_and_look.ipynb
```

### What the Notebook Does (in order)

1. Loads and explores the DGA dataset (shape, class distribution, gas statistics)
2. Runs all three classical methods and computes their accuracy on the test set
3. Engineers 9 features from the raw 5 gases
4. Trains XGBoost with stratified split and class balancing
5. Applies isotonic calibration and computes ECE before/after
6. Generates SHAP global and local explanations
7. Produces the fleet risk ranking CSV and chart

Running all cells takes approximately 2–3 minutes on a standard laptop.

---

## 11. Dataset

**Source:** IEEE DataPort — Dissolved Gas Analysis (DGA) Dataset
**DOI:** [10.21227/27vy-h479](https://ieee-dataport.org/documents/dissolved-gas-analysis-dga-dataset)
**Size:** 409 samples, 7 fault classes, 5 gas features
**License:** CC BY 4.0

The dataset is not committed to this repository due to its third-party license. Download it directly from IEEE DataPort and place the CSV in the `data/` folder before running the notebook.

**Known limitation:** 41 of 49 IEC TC10 reference rows are present in this dataset's training split. This leakage has been identified, documented, and will be corrected in Part 2.

---

## 12. Standards & References

**International Standards:**

- **IEC 60599:2022** — Mineral oil-impregnated electrical equipment in service — Guide to the interpretation of dissolved and free gases analysis. International Electrotechnical Commission.
- **IEEE C57.104-2019** — IEEE Guide for the Interpretation of Gases Generated in Mineral Oil-Immersed Transformers. IEEE Power and Energy Society.
- **CIGRE Technical Brochure 761 (2019)** — Advances in DGA interpretation. Working Group A2.47.

**Academic References:**

- Sutikno, T., et al. (2024). Multi-method DGA interpretation for power transformer fault diagnosis. *Heliyon*, 10(4), e25975. — Documents the disagreement between classical methods on the same sample; motivates the consensus engine built in Part 2.
- Kapoor, A., & Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. *Patterns*, 4(9). — Taxonomy for data leakage types used to characterize the IEC TC10 duplication issue.
- Duval, M., & Lamarre, L. (2014). The Duval Pentagon — A New Improved Method for the Interpretation of Dissolved Gas Analysis in Transformers. *IEEE Electrical Insulation Magazine*, 30(6), 9–12.

---

## 13. Roadmap — Part 2

Part 2 extends this pipeline from a working academic model toward an industrial-grade, deployable system:

| Step | Enhancement | Purpose |
|------|-------------|---------|
| 2 | Multi-method Consensus Engine | Rogers Ratio + Doernenburg + Duval Pentagon voting alongside ML |
| 3 | Abstention Layer | Model says "refer to engineer" when confidence < threshold |
| 4 | Leakage Audit & Clean Holdout | Remove IEC TC10 duplicates; rebuild honest test set |
| 5 | Deeper Model Comparison | SVM, LightGBM, Random Forest with bootstrap confidence intervals |
| 6 | Engineer-Facing SHAP | Plain-language gas narratives for non-ML operators |
| 7 | Automated Per-Transformer Reports | HTML report generated automatically for each unit |
| 8 | Gas Plausibility Validation | Physical bounds checking before any diagnosis is attempted |
| 9 | Full Reproducibility Package | Pinned environment, run.py, GitHub CI, tagged release |
| 10 | pydga Python Library | Package and publish to PyPI for community use |

---

## 14. About

**Project by:** Ankit Raj
**Institution:** KIIT University, Bhubaneswar — B.Tech, Electrical Engineering (Final Year)
**Contact:** [rankit0714@gmail.com](mailto:rankit0714@gmail.com)
**GitHub:** [github.com/AR0714](https://github.com/AR0714)

This project began as a final-year academic exercise in applying machine learning to power systems condition monitoring. It is being developed progressively toward an industrial-quality, open-source tool for the transformer diagnostics community — following IEEE and IEC standards throughout.

Contributions, issues, and suggestions are welcome. If you work in transformer maintenance, power utility operations, or electrical asset management and find this useful or have real-world DGA data to contribute, please open an issue or reach out directly.

---

*Built with Python · XGBoost · SHAP · scikit-learn · Pandas · Matplotlib*
*Guided by IEC 60599 · IEEE C57.104 · CIGRE TB 761*

---
