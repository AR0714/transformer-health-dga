# DECISIONS.md — Intelligent Transformer Health Monitoring Using DGA
# Complete Project Record: Every Step, File, Technology, and Decision

Author : Ankit Raj (rankit0714@gmail.com), Final-Year EE, KIIT University
Repo   : https://github.com/AR0714/transformer-health-dga
Updated: 2024 (Part 4-A complete; Part 4-B in progress)

---

## SECTION 1 — PROJECT OVERVIEW

Goal : Build an end-to-end transformer health monitoring system using
       Dissolved Gas Analysis (DGA) with an XGBoost classifier, RAG-powered
       knowledge base, local Ollama LLM, and a live Flask dashboard.

Standard : IEC 60599 (interpretation of DGA data for fault diagnosis)

Fault Classes (7):
  Normal | PD (Partial Discharge) | D1 (Low-energy discharge) |
  D2 (High-energy discharge) | T1 (<300 C thermal) |
  T2 (300-700 C thermal) | T3 (>700 C thermal)

Monitored Gases (5): H2, CH4, C2H6, C2H4, C2H2 (in ppm)

Derived Ratios (4, appended as features):
  - C2H2/(C2H4 + eps)   — Rogers ratio 1
  - CH4/(H2 + eps)       — Rogers ratio 2
  - C2H4/(C2H6 + eps)   — Rogers ratio 3
  - total_gas = sum of all 5 gases

Full Feature Vector (9): [H2, CH4, C2H6, C2H4, C2H2, r1, r2, r3, total_gas]

---

## SECTION 2 — COMPLETE STEP LOG

### PART 1 — Data & Feature Engineering (Steps 1–8)

Step 1 — Project setup
  Created directory structure: data/, models/, src/, app/, notebooks/,
  reports/, data/iec_knowledge_base/
  Initialised Git repo; created .gitignore (excludes *.joblib, __pycache__)

Step 2 — Dataset creation (synthetic IEC 60599 data)
  File : data/dga_dataset.csv
  Method: NumPy-based generator seeded with IEC 60599 typical gas ranges.
  Rows  : ~3 500 samples, balanced across 7 fault classes (~500/class).
  Columns: H2, CH4, C2H6, C2H4, C2H2, label

Step 3 — Exploratory Data Analysis (EDA)
  Notebook: notebooks/01_eda.ipynb
  Plots: class distribution bar chart, pairplot of 5 gases, correlation heatmap.
  Finding: C2H2 is the strongest single discriminator for discharge faults.

Step 4 — Feature Engineering
  Added 4 derived ratio columns to dga_dataset.csv (r1, r2, r3, total_gas).
  Validated no NaN/Inf values after ratio computation (eps=1e-6 guard).

Step 5 — Train/test split
  80/20 stratified split; random_state=42; saved split indices for
  reproducibility. Confirmed class balance preserved in both splits.

Step 6 — Baseline model exploration
  Tested: Random Forest, Decision Tree, Logistic Regression (sklearn).
  Result: RF reached ~87% test accuracy; LogReg ~73%.
  Decision: Proceed with XGBoost for its gradient-boosted precision.

Step 7 — Label encoding
  LabelEncoder fitted on 7 class names; saved as models/label_encoder.joblib.
  Mapping: D1=0, D2=1, Normal=2, PD=3, T1=4, T2=5, T3=6

Step 8 — Data validation script
  src/validate_data.py checks: no missing values, correct column names,
  label set exactly matches 7 classes, ratio columns in expected range.

### PART 2 — Model Training & Calibration (Steps 9–16)

Step 9 — XGBoost training
  File   : models/xgb_dga_base.joblib
  Params : n_estimators=200, max_depth=6, learning_rate=0.1,
           use_label_encoder=False, eval_metric='mlogloss',
           random_state=42
  Result : Test accuracy 94.3%, macro F1 0.943

Step 10 — Cross-validation
  5-fold stratified CV on full dataset; mean accuracy 93.8% +/- 1.2%.
  Confirmed no overfitting (train accuracy 97.1%).

Step 11 — Confusion matrix analysis
  reports/confusion_matrix.png
  Worst confusion: T1 vs Normal (similar gas levels at borderline).
  Action: accepted — real-world DGA data has this ambiguity.

Step 12 — Probability calibration — Temperature Scaling
  Reason : Raw XGBoost softmax probabilities are over-confident (ECE=0.158).
  Method : Temperature Scaling — single scalar T* learned by minimising NLL
           on validation set using scipy.optimize.minimize_scalar.
  Result : T* = 1.7334; calibrated ECE = 0.106 (33% improvement).
  File   : models/xgb_dga_calibrated.joblib

Step 13 — CalibratedDGAModel wrapper class
  src/calibrated_model.py
  Wraps base XGBoost + T* scalar.
  Public API:
    predict_sample(H2, CH4, C2H6, C2H4, C2H2) -> dict:
      {"diagnosis": str, "confidence": float (0-1),
       "probabilities": dict[class->float], "reliability": str}
  reliability thresholds: >=0.85 -> "High", >=0.60 -> "Medium", else "Low"

Step 14 — Reliability tagging
  Added to predict_sample() output.
  Enables chatbot to say "I'm highly confident this is a T3 fault"
  vs "I'm not very sure, please run a follow-up test."

Step 15 — Model evaluation report
  reports/model_evaluation.txt
  Contains: accuracy, per-class precision/recall/F1, confusion matrix counts,
  ECE before/after calibration, Temperature Scaling parameter.

Step 16 — Notebook for full training pipeline
  notebooks/02_model_training.ipynb
  Reproducible end-to-end: load data -> engineer features -> train XGBoost ->
  calibrate -> save .joblib files -> print evaluation report.

### PART 3 — RAG Pipeline + Ollama Agent (Steps 17–19)

Step 17 — IEC 60599 knowledge base (9 text files)
  Directory: data/iec_knowledge_base/
  Files:
    sec_normal.txt   — characteristics of normal transformer operation
    sec_pd.txt       — partial discharge gas signatures and causes
    sec_d1.txt       — low-energy discharge description
    sec_d2.txt       — high-energy discharge description
    sec_t1.txt       — thermal fault <300 C
    sec_t2.txt       — thermal fault 300-700 C
    sec_t3.txt       — thermal fault >700 C
    sec_ratios.txt   — explanation of Rogers/Doernenburg ratio method
    sec_action.txt   — IEC 60599 action levels and recommended responses
  Total: ~3 000 words of IEC-aligned technical content.

Step 18 — FAISS vector index
  src/build_kb.py
    - Loads all 9 txt files; chunks at ~120 words with 20-word overlap.
    - Encodes with sentence-transformers all-MiniLM-L6-v2 (384-dim embeddings).
    - L2-normalises vectors so inner product = cosine similarity.
    - Builds FAISS IndexFlatIP; saves to models/faiss_index/index.faiss.
    - Saves chunk metadata (source file, chunk text) to models/faiss_index/metadata.json.
  Result: 11 chunks indexed.

  src/rag_engine.py
  Class RAGEngine:
    retrieve(query, k=3) -> list[dict]  # top-k chunks by cosine similarity
    format_context(chunks) -> str       # formats for prompt injection

Step 19 — Ollama agent CLI
  Technology: Ollama (local LLM runtime, free, no API key required)
  Model     : llama3.2 (2.0 GB, pulled via `ollama pull llama3.2`)
  Endpoint  : http://localhost:11434/api/generate

  src/agent.py
    query_ollama(prompt, model="llama3.2") -> str
    build_prompt(diagnosis_dict, context_str, user_question) -> str

  src/diagnose_agent.py  — interactive CLI loop:
    1. Read sensor (TransformerSensor or manual input)
    2. Run predict_sample() -> diagnosis
    3. Retrieve top-3 IEC chunks via RAG
    4. Build prompt: system + context + diagnosis + user question
    5. Send to Ollama; stream reply to console

  Git tag: v2.0.0 — Part 3 complete

### PART 4 PHASE A — Conversational Chatbot Backend (Steps 20–24)

Step 20 — TransformerSensor (real-time simulator)
  src/sensor_simulator.py
  Class TransformerSensor:
    __init__(transformer_type, primary_kv, secondary_v, rated_mva,
             fault_scenario, base_hours)
    _noise(value, pct=0.05)     — Gaussian noise (5-8%)
    _load_pct()                  — sinusoidal daily load profile (0.6-1.0)
    read()                       — returns full sensor dict:
      {transformer_type, primary_kv, secondary_v, rated_mva, load_pct,
       temperature_C, gases:{H2,CH4,C2H6,C2H4,C2H2}, fault_scenario,
       timestamp}
    read_gases_only()            — convenience wrapper

  BASE_GASES per fault_scenario (ppm, before noise):
    Normal: H2=20,  CH4=8,   C2H6=4,   C2H4=2,   C2H2=0
    PD    : H2=800, CH4=20,  C2H6=5,   C2H4=3,   C2H2=1
    D1    : H2=300, CH4=30,  C2H6=10,  C2H4=20,  C2H2=50
    D2    : H2=500, CH4=50,  C2H6=10,  C2H4=200, C2H2=400
    T1    : H2=60,  CH4=300, C2H6=150, C2H4=50,  C2H2=2
    T2    : H2=80,  CH4=500, C2H6=100, C2H4=300, C2H2=5
    T3    : H2=100, CH4=800, C2H6=100, C2H4=600, C2H2=10

Step 21 — ConversationMemory (sliding window)
  src/transformer_memory.py
  Class ConversationMemory:
    __init__(max_turns=8)        — deque(maxlen=16) for 8 user+8 assistant turns
    add_user(text)
    add_assistant(text)
    format_for_prompt() -> str   — returns "User: ...
Assistant: ..." history
    last_diagnosis() -> str|None — scans history for fault class keywords

Step 22 — IntentDetector (9-class keyword router)
  src/intent_detector.py
  9 Intent classes:
    health_status    — "okay","ok","fine","healthy","health","status","worried",...
    voltage_query    — "voltage","volt","kv","primary","secondary",...
    gas_query        — "gas","gases","h2","hydrogen","ch4","dga","ppm",...
    temperature_query— "temperature","temp","hot","heat","thermal",...
    load_query       — "load","loading","capacity","overload","overloaded","usage",...
    action_advice    — "should i","what should","action","recommend","shutdown",...
    life_estimate    — "life","lifetime","how long","years","left","remaining",...
    explain_fault    — "explain","what is","why","cause","d1","d2","t1","t2","t3",...
    general_chat     — catch-all (no patterns)

  Method: regex word-boundary matching () so "overloaded" matches "overload"
          pattern; words checked individually so "how many years does it have
          left" matches "years" and "left".

  Patches applied during testing:
    - Added "overloaded" to load_query patterns
    - Added "years","left" to life_estimate patterns

Step 23 — TransformerChatbot (main orchestrator)
  src/transformer_chatbot.py
  Class TransformerChatbot:
    __init__(fault_scenario="Normal"):
      - Loads models/xgb_dga_calibrated.joblib via joblib.load()
      - Instantiates RAGEngine, ConversationMemory(8), TransformerSensor
    _diagnose(reading) -> dict:
      - Extracts g = reading["gases"]
      - Calls clf.predict_sample(g["H2"],g["CH4"],g["C2H6"],g["C2H4"],g["C2H2"])
      - Returns {diagnosis, confidence (%), probabilities}
    chat(user_text) -> str:
      1. memory.add_user(user_text)
      2. sensor.read() -> reading
      3. _diagnose(reading) -> diag
      4. detect_intent(user_text) -> intent
      5. Route to handler:
           health_status    -> _handle_health_status()  [rule-based]
           voltage_query    -> _handle_voltage()        [rule-based]
           gas_query        -> _handle_gas()            [Ollama + RAG]
           temperature_query-> _handle_temperature()   [rule-based]
           load_query       -> _handle_load()           [rule-based]
           action_advice    -> _handle_action()         [Ollama + RAG]
           life_estimate    -> _handle_life()           [rule-based formula]
           explain_fault    -> _handle_explain()        [Ollama + RAG]
           general_chat     -> _handle_general()        [Ollama + memory]
      6. memory.add_assistant(reply)
      7. return reply

  Bugs fixed during development:
    - FileNotFoundError: model filename was calibrated_dga_model.pkl;
      corrected to xgb_dga_calibrated.joblib
    - AttributeError: 'CalibratedDGAModel' has no 'classes_';
      fixed by using predict_sample() return dict instead
    - TypeError: predict_sample() missing 4 positional args;
      fixed by passing 5 individual gas values, not a dict

Step 24 — Chat CLI evaluation script
  src/chat_cli.py
  Runs 10 test questions across all 7 fault scenarios (70 total calls).
  All 10 question types answered correctly:
    1. "Is the transformer okay?"         -> health_status
    2. "What is the voltage?"             -> voltage_query
    3. "What are the gas levels?"         -> gas_query
    4. "Is it overheating?"              -> temperature_query
    5. "Is the load too high?"           -> load_query
    6. "What action should I take?"      -> action_advice
    7. "How many years does it have left?"-> life_estimate
    8. "Explain what T3 fault means"     -> explain_fault
    9. "Tell me about DGA"               -> explain_fault
   10. "Hello, how are you?"             -> general_chat

  Git tag: v3.0.0-alpha — Part 4-A complete

---

## SECTION 3 — ALL FILES CREATED

### Source code (src/)
  src/validate_data.py         — dataset integrity checker
  src/calibrated_model.py      — CalibratedDGAModel wrapper class
  src/build_kb.py              — builds FAISS index from IEC txt files
  src/rag_engine.py            — RAGEngine: retrieve + format_context
  src/agent.py                 — query_ollama(), build_prompt()
  src/diagnose_agent.py        — interactive CLI (Part 3)
  src/sensor_simulator.py      — TransformerSensor (Step 20)
  src/transformer_memory.py    — ConversationMemory (Step 21)
  src/intent_detector.py       — IntentDetector 9-class (Step 22)
  src/transformer_chatbot.py   — TransformerChatbot orchestrator (Step 23)
  src/chat_cli.py              — evaluation CLI (Step 24)

### Data (data/)
  data/dga_dataset.csv                          — 3 500-row synthetic DGA dataset
  data/iec_knowledge_base/sec_normal.txt
  data/iec_knowledge_base/sec_pd.txt
  data/iec_knowledge_base/sec_d1.txt
  data/iec_knowledge_base/sec_d2.txt
  data/iec_knowledge_base/sec_t1.txt
  data/iec_knowledge_base/sec_t2.txt
  data/iec_knowledge_base/sec_t3.txt
  data/iec_knowledge_base/sec_ratios.txt
  data/iec_knowledge_base/sec_action.txt

### Models (models/)
  models/label_encoder.joblib          — sklearn LabelEncoder (7 classes)
  models/xgb_dga_base.joblib           — base XGBoost classifier
  models/xgb_dga_calibrated.joblib     — CalibratedDGAModel (T*=1.7334)
  models/faiss_index/index.faiss       — FAISS IndexFlatIP (11 chunks, 384-dim)
  models/faiss_index/metadata.json     — chunk text + source file mapping

### Notebooks (notebooks/)
  notebooks/01_eda.ipynb               — EDA: distributions, pairplot, heatmap
  notebooks/02_model_training.ipynb    — full training + calibration pipeline

### Reports (reports/)
  reports/confusion_matrix.png         — 7x7 confusion matrix heatmap
  reports/model_evaluation.txt         — accuracy, F1, ECE before/after

### App (app/) — Part 4-B pending
  app/server.py                        — Flask REST API [PENDING Step 25]
  app/static/index.html               — dashboard UI  [PENDING Step 26]
  app/static/style.css                — dashboard CSS  [PENDING Step 26]
  app/static/app.js                   — dashboard JS   [PENDING Step 26]

### Root
  README.md                            — project description, setup, usage
  DECISIONS.md                         — this file
  requirements.txt                     — Python dependencies
  .gitignore                           — excludes *.joblib, __pycache__, .env

---

## SECTION 4 — TECHNOLOGIES & LIBRARIES

### Machine Learning
  xgboost==1.7.x        — gradient-boosted classifier (7-class DGA diagnosis)
  scikit-learn          — LabelEncoder, train_test_split, cross_val_score,
                          classification_report, confusion_matrix
  scipy                 — minimize_scalar for Temperature Scaling optimisation
  numpy                 — feature engineering, noise generation, ratio computation
  pandas                — CSV loading, feature matrix construction
  joblib                — model serialisation (.joblib files)
  matplotlib, seaborn   — EDA plots, confusion matrix heatmap

### RAG / Embeddings / Vector Search
  sentence-transformers — all-MiniLM-L6-v2 model (384-dim embeddings)
  faiss-cpu             — IndexFlatIP (exact cosine search via L2-normalised IP)

### LLM
  Ollama                — local LLM runtime (free, no API key, runs on CPU/GPU)
  llama3.2              — 2.0 GB model pulled via `ollama pull llama3.2`
  requests              — HTTP calls to Ollama API at localhost:11434

### Backend (planned)
  Flask                 — REST API server (app/server.py, Step 25)
  flask-cors            — CORS headers for frontend JS fetch calls

### Frontend (planned)
  Vanilla HTML/CSS/JS   — no React, no Vue, no Streamlit
  SVG animation         — animated transformer diagram in dashboard
  CSS gauge widgets     — live gauges for gas levels, temperature, load

### Deployment (planned)
  Render.com free tier  — web service hosting (Step 27)
  gunicorn              — WSGI server for Flask on Render

### Environment
  Python 3.10+          — Windows (cp1252 encoding) and cloud CI
  Jupyter Notebook      — development and step-by-step execution
  Git + GitHub          — version control, releases

---

## SECTION 5 — ARCHITECTURAL DECISIONS

### Decision 1 — XGBoost over Neural Networks
Why: Tabular data with 9 features; XGBoost outperforms NNs on small tabular
     datasets (94.3% vs ~89% for MLP in initial tests).
     Training takes seconds; no GPU required; interpretable feature importances.

### Decision 2 — 9 Input Features (5 gases + 4 derived ratios)
Why: IEC 60599 defines Rogers ratios as primary diagnostic method.
     Ratios encode relative gas relationships; improves F1 by ~3% vs raw gases.
     total_gas catches overall severity.

### Decision 3 — 7 Fault Classes per IEC 60599
Why: Industry-standard classification. 6 fault types + Normal covers all
     common transformer failure modes. Matches the IEC 60599 Annex A table.

### Decision 4 — Temperature Scaling for Calibration
Why: Raw XGBoost probabilities are over-confident (ECE=0.158).
     Temperature Scaling is the simplest post-hoc calibration method;
     single parameter T*; no additional training data required.
Result: ECE improved from 0.158 -> 0.106 (T* = 1.7334).

### Decision 5 — CalibratedDGAModel Wrapper
Why: Encapsulates base model + T* scalar + feature engineering in one object.
     predict_sample(H2,CH4,C2H6,C2H4,C2H2) is the only public API needed.
     Prevents caller from needing to know internal feature order.

### Decision 6 — Ollama (Local LLM) over Cloud APIs
Why: No API key, no cost, works offline, GDPR-friendly (data stays local).
     llama3.2 (2 GB) sufficient for technical explanation tasks.
     Tradeoff: slower on CPU-only machines, requires Ollama running.

### Decision 7 — FAISS + sentence-transformers for RAG
Why: FAISS is the fastest open-source vector search library.
     all-MiniLM-L6-v2 is small (80 MB), fast, and accurate for semantic search.
     IndexFlatIP with L2 normalisation gives exact cosine similarity.

### Decision 8 — 9 IEC KB Files, ~120-word Chunks with 20-word Overlap
Why: 9 files = one per fault class + ratios + action levels.
     120-word chunks fit in LLM context without losing coherence.
     20-word overlap prevents splitting mid-sentence at fault boundaries.

### Decision 9 — Intent-Based Routing (Not Pure LLM)
Why: Pure LLM routing is unpredictable; keyword regex gives 100% determinism
     for known question types. Only complex explanations need Ollama.
     Faster response for simple operational queries (voltage, load, temperature).

### Decision 10 — 8-Turn Sliding Window Memory
Why: 16-entry deque (8 user + 8 assistant) = ~2 000-3 000 tokens of context.
     Enough to maintain multi-turn diagnostic conversations.
     Prevents unbounded memory growth in long sessions.

### Decision 11 — Real-Time Sensor Simulation
Why: No real transformer available for testing.
     Sinusoidal daily load profile mimics real load variation.
     Per-fault BASE_GASES + 5-8% Gaussian noise gives realistic, varied readings.

### Decision 12 — Flask + Vanilla JS (No React, No Streamlit)
Why: Streamlit rejected (user preference). React adds build complexity.
     Flask + plain JS is lightweight, deployable on free tier (Render.com).
     Single HTML file with inline SVG transformer diagram is zero-dependency.

### Decision 13 — Render.com Free Tier Deployment
Why: Free, no credit card for basic tier, supports Flask + gunicorn.
     Automatic deploys from GitHub main branch.
     Tradeoff: spins down after 15 min inactivity (acceptable for demo).

---

## SECTION 6 — GIT RELEASE HISTORY

v1.0.0   — Part 1 + 2 complete
           Dataset, EDA, XGBoost training, Temperature Scaling calibration.
           Files: data/dga_dataset.csv, models/*.joblib, notebooks/*.ipynb

v2.0.0   — Part 3 complete
           IEC knowledge base, FAISS index, RAG engine, Ollama agent CLI.
           Files: data/iec_knowledge_base/, models/faiss_index/, src/rag_engine.py,
                  src/agent.py, src/diagnose_agent.py

v3.0.0-alpha — Part 4-A complete
           Sensor simulator, conversation memory, intent detector,
           TransformerChatbot, chat CLI evaluation.
           Files: src/sensor_simulator.py, src/transformer_memory.py,
                  src/intent_detector.py, src/transformer_chatbot.py,
                  src/chat_cli.py

v3.0.0   — Part 4-B target (Steps 25-27, in progress)
           Flask REST API, live dashboard UI, Render.com deployment.

---

## SECTION 7 — REMAINING WORK (Part 4-B)

Step 25 — app/server.py: Flask REST API
  Endpoints:
    GET  /api/sensors           — returns latest sensor reading (JSON)
    POST /api/chat              — body: {message, fault_scenario}; returns reply
    POST /api/diagnosis         — body: {H2, CH4, C2H6, C2H4, C2H2}; returns diagnosis

Step 26 — app/static/: Dashboard UI
  index.html: animated SVG transformer schematic + tabbed panel
  Live gas level gauges (refresh every 5 s via setInterval + fetch /api/sensors)
  Chatbot panel: message input, scrollable chat history
  style.css, app.js (vanilla JS, no framework)

Step 27 — Render.com deployment + final commit
  Procfile: web: gunicorn app.server:app
  requirements.txt updated with flask, flask-cors, gunicorn
  README.md: add live demo URL + architecture diagram
  LinkedIn post drafted
  Final Git tag: v3.0.0
