# Intelligent Transformer Health Monitoring using DGA & ML

> Classifies 7 internal fault types in oil-filled power transformers from
> Dissolved Gas Analysis — beating the classical Duval Triangle baseline
> by +23 percentage points using calibrated XGBoost, physics-informed
> IEC 60599 feature engineering, and a RAG-powered LLM diagnostic agent.

**Student:** Ankit Raj | B.Tech Electrical Engineering (Final Year) | KIIT University, Bhubaneswar
**Standards:** IEC 60599:2022 · IEEE C57.104-2019 · CIGRE TB 761
**Dataset:** IEEE DataPort — DOI: 10.21227/27vy-h479

---

## Results

| Method | Accuracy (70-row sealed test) | Notes |
|--------|-------------------------------|-------|
| Key Gas (classical) | ~60% | Single dominant gas only |
| IEC 60599 Ratios (classical) | Partial | No-decision on ~30% of cases |
| Duval Triangle (classical) | 57.1% | Industry standard since 1970s |
| Random Forest (ML baseline) | 78.6% | With physics-informed features |
| **XGBoost — calibrated (ours)** | **80.0%** | **ECE: 0.15 → 0.10 after calibration** |

**+23 percentage points above the Duval Triangle.**
Both methods evaluated on the same 70-row held-out test set (10 samples
per class, stratified split).

---

## What This Project Does

A power transformer costs USD 2–10 million and takes 18–24 months to
replace. When faults develop internally — overheating, partial discharge,
arcing — the insulating oil breaks down and releases specific dissolved
gases. This project automates the interpretation of those gases across
four parts:

**Part 1:** Classical methods + calibrated XGBoost classifier
**Part 2:** Probability calibration, SHAP explainability, fleet risk ranking
**Part 3:** RAG-powered LLM diagnostic agent (IEC 60599 knowledge base + Ollama)
**Part 4:** Live web dashboard with conversational chatbot and sensor simulator

**The pipeline:**

Raw DGA Gases (5 key gases: H₂ CH₄ C₂H₆ C₂H₄ C₂H₂)
↓
Classical Methods → Key Gas · IEC 60599 Ratios · Duval Triangle
↓
Physics-Informed Feature Engineering (~13 features)
IEC gas ratios + Duval coordinates + gas percentage fractions
↓
Calibrated XGBoost Classifier
↓
SHAP Explainability → Top fault drivers identified per prediction
↓
Fleet Risk Ranking → Worst units surfaced first
↓
RAG Engine (FAISS + sentence-transformers + IEC 60599 knowledge base)
↓
LLM Agent (Ollama / llama3.2) → Plain-language diagnostic answers
↓
Live Web Dashboard + Conversational Chatbot


---

## The 7 Fault Classes (IEC 60599)

| Class | Fault | Key Gas Signature | Severity |
|-------|-------|-------------------|----------|
| Normal | No fault | All gases typical | ✅ Safe |
| PD | Partial Discharge | H₂ dominant | ⚠️ Monitor |
| T1 | Thermal < 300°C | CH₄ + C₂H₆ dominant | ⚠️ Caution |
| T2 | Thermal 300–700°C | C₂H₄ dominant | 🔶 Warning |
| T3 | Thermal > 700°C | C₂H₄ critically high | 🔴 Critical |
| D1 | Low-energy discharge | H₂ + low C₂H₂ | 🔶 Warning |
| D2 | High-energy arcing | C₂H₂ > 50 ppm | 🚨 Emergency |

The ML model uses 5 key DGA gases (H₂ CH₄ C₂H₆ C₂H₄ C₂H₂).
The dashboard additionally monitors CO and CO₂ for paper degradation context.

---

## Key Technical Findings

**1. Physics-informed features beat raw gas values alone.**
Feature engineering adds ~13 inputs from 5 raw gases: three IEC 60599
ratio features (C₂H₂/C₂H₄, CH₄/H₂, C₂H₄/C₂H₆), Duval Triangle X/Y
coordinates (barycentric projection), and gas percentage fractions. These
encode decades of electrochemical physics that a data-only model would
need thousands of samples to learn implicitly.

**2. Calibration matters more than accuracy for risk ranking.**
Raw XGBoost outputs overconfident probabilities (Expected Calibration
Error = 0.15). After isotonic regression calibration via
CalibratedClassifierCV, ECE dropped to 0.10 — a 33% reduction. A fleet
ranking built on uncalibrated probabilities produces an unreliable
priority list. After calibration, when the model says 80% confident it
is correct approximately 80% of the time.

**3. SHAP confirmed the model learned real transformer chemistry.**
For D2 (high-energy arcing) predictions, C₂H₂ (acetylene) has SHAP
value +0.74 — the highest of all features. This validates the model
against IEC 60599 physics: acetylene is produced exclusively by
electrical arcing above 1000°C and is the primary internationally
recognised marker for D2. The model learned what engineers already know.

**4. The Duval Triangle's blind spot is the discharge-thermal boundary.**
Confusion matrix analysis shows the model's remaining errors cluster at
the PD↔D1 and D1↔D2 boundary — physically sensible, because these
faults share overlapping gas chemistry at boundary concentrations.

---

## System Architecture

### Part 1–2: ML Pipeline (`notebooks/01_load_and_look.ipynb`)
- Dataset: ~584 training rows (after deduplication), 70-row sealed test
- Stratified 80/20 split (preserves rare T2 class: 21 training samples)
- Feature engineering: ~13 physics-informed features from 5 raw gases
- Model: XGBoost (300 estimators, max_depth=4, learning_rate=0.1)
- Calibration: isotonic regression via CalibratedClassifierCV
- Evaluation: 5-fold stratified cross-validation, macro-F1 primary metric
- Explainability: SHAP TreeExplainer (summary plot + per-prediction waterfall)

### Classical Methods (`src/classical/`)
- `classical_methods.py` — Key Gas, IEC 60599 ratios, Duval Triangle
- `key_gas.py` — dominant-gas fault indicator
- `iec_ratios.py` — three-ratio IEC 60599 classifier with no-decision handling
- `duval.py` — Duval Triangle using matplotlib.path polygon zone test

### Part 3: RAG Diagnostic Agent (`src/`)
- `build_kb.py` — builds FAISS vector index from IEC 60599 / IEEE text summaries
- `rag_engine.py` — semantic retrieval engine (sentence-transformers + FAISS)
- `agent.py` — Ollama HTTP integration (llama3.2, localhost:11434)
- `diagnose_agent.py` — CLI agent: gas input → XGBoost → SHAP → IEC retrieval → LLM response

### Part 4: Web Dashboard (`app/`)
- `server.py` — Flask server with three endpoints:
  - `GET /api/sensors` — live DGA simulation (7 gases, ±4% jitter, 3s polling)
  - `GET /api/diagnosis` — ML classification on current sensor readings (5s polling)
  - `POST /api/chat` — conversational chatbot with live sensor context injection
- `static/index.html` — HTML/CSS/JS dashboard with:
  - Animated transformer SVG (core color: green/yellow/red by fault severity)
  - Live gas-level bars with IEC 60599 threshold markers
  - Health status badge and confidence meter
  - Embedded conversational chatbot panel

### Part 4 Backend (`src/`)
- `sensor_simulator.py` — realistic DGA + electrical reading simulator
- `transformer_memory.py` — 8-turn conversation memory with sensor snapshots
- `intent_detector.py` — classifies user question type (health/voltage/gas/advice)
- `transformer_chatbot.py` — orchestrator: intent → sensor → model → RAG → LLM
- `chat_cli.py` — CLI chatbot for testing before dashboard

---

## Honest Limitations

This section exists because engineering credibility requires it.

- **Test set is curated.** The 70-row sealed test uses 10 samples per class,
  not a natural distribution. Real-world performance on imbalanced fleet
  data will differ, particularly for rare fault types.

- **Data leakage in IEC TC10 benchmark.** 41 of 49 IEC TC10 reference
  rows (84%) overlap with training data. Any accuracy reported on all
  49 IEC TC10 rows is inflated. The honest primary metric is the
  independent 70-row sealed test. The 8 genuinely unseen IEC TC10 rows
  are too few for strong statistical claims.

- **T2 class has 21 training samples.** Per-class recall for T2 is
  less stable than for other classes. Treat T2 predictions with
  additional caution.

- **DGA is a snapshot, not a time-series.** IEC standards weight gas
  trends heavily; this project uses single-snapshot readings only.
  Time-series trend features would improve performance but require
  longitudinal data not available in the current dataset.

- **Not validated on physical transformers.** Results are on a benchmark
  dataset. Field validation on an instrumented transformer would be
  required before operational deployment.

- **LLM runs locally via Ollama.** The cloud-deployed version switches
  the LLM endpoint to Groq / Together AI. No change to chatbot logic —
  only the HTTP endpoint changes.

---

## How to Run

### Requirements
```bash
pip install -r requirements.txt
# Key packages: xgboost scikit-learn shap pandas numpy flask
#               sentence-transformers faiss-cpu joblib requests
```

### Local LLM (One-Time Setup)
```bash
# Install Ollama from ollama.com
ollama pull llama3.2   # ~2 GB download, runs on 8 GB RAM
ollama serve           # Start in a separate terminal
```

### Run the Web Dashboard
```bash
python app/server.py
# Open http://127.0.0.1:5000
```

### Run the CLI Diagnostic Agent (Part 3)
```bash
python src/diagnose_agent.py
```

### Run the Conversational Chatbot (Part 4 CLI)
```bash
python src/chat_cli.py
```

### Run the ML Notebook
```bash
jupyter lab
# Open notebooks/01_load_and_look.ipynb
```

### Dataset
Download from IEEE DataPort: `https://ieee-dataport.org/documents/dissolved-gas-analysis-dga`
Place files in `data/` directory (gitignored).

---

## Standards Referenced

- **IEC 60599:2022** — Mineral oil-filled electrical equipment:
  interpretation of dissolved and free gases analysis
- **IEEE C57.104-2019** — Guide for the Interpretation of Gases
  Generated in Mineral Oil-Immersed Transformers
- **CIGRE Technical Brochure 761** — Advances in DGA interpretation

---

## Repository Structure

transformer-health-dga/
├── data/
│ ├── raw/ # IEEE DataPort DGA dataset
│ └── iec_knowledge_base/ # IEC 60599 text files for RAG
├── models/
│ ├── xgb_dga_calibrated.joblib # Trained calibrated model
│ └── faiss_index/ # FAISS vector index (Part 3)
│ ├── index.faiss
│ └── metadata.json
├── notebooks/
│ └── 01_load_and_look.ipynb # Full ML pipeline (Parts 1–2)
├── src/
│ ├── classical/
│ │ ├── classical_methods.py # Key Gas, IEC ratios, Duval
│ │ ├── key_gas.py
│ │ ├── iec_ratios.py
│ │ └── duval.py
│ ├── calibration.py # CalibratedDGAModel class
│ ├── predict.py # Prediction wrapper
│ ├── build_kb.py # Builds FAISS knowledge base
│ ├── rag_engine.py # RAG retrieval engine
│ ├── agent.py # Ollama LLM integration
│ ├── diagnose_agent.py # CLI diagnostic agent
│ ├── sensor_simulator.py # Live sensor simulation
│ ├── transformer_memory.py # Conversation memory
│ ├── intent_detector.py # Question intent classifier
│ ├── transformer_chatbot.py # Conversational orchestrator
│ └── chat_cli.py # CLI chatbot interface
├── app/
│ ├── server.py # Flask web server
│ └── static/
│ ├── index.html # Dashboard page
│ ├── transformer.css # Styles + animations
│ └── dashboard.js # Live polling + chatbot JS
├── reports/
│ ├── shap_summary_bar.png
│ ├── calibration_curve.png
│ └── confusion_matrix.png
├── DECISIONS_3.md # All key engineering decisions
├── requirements.txt
└── README.md


---

## About

Built as a final-year B.Tech Electrical Engineering project at KIIT University.
The project sits at the intersection of power systems domain knowledge
(IEC/IEEE standards, DGA physics) and modern ML practice (calibration,
explainability, fleet prioritisation, RAG-based LLM agents).

**Author:** Ankit Raj
**Contact:** ankitforward47@gmail.com
**LinkedIn:** linkedin.com/in/ankitraj0714
**GitHub:** github.com/AR0714

Run all cells top to bottom.

## Fault Classes
Normal | PD | D1 | D2 | T1 | T2 | T3

## Author
Ankit Raj — KIIT University, 2026
