# DECISIONS.md -- Intelligent Transformer Health Monitoring Using DGA
# Complete Project Record: Every Step, File, Technology, and Decision

Author : Ankit Raj (rankit0714@gmail.com), Final-Year EE, KIIT University
Repo   : https://github.com/AR0714/transformer-health-dga

---

## SECTION 1 -- PROJECT OVERVIEW

Goal : End-to-end transformer health monitoring using Dissolved Gas Analysis
       (DGA) with XGBoost classifier, RAG knowledge base, local Ollama LLM,
       and a live Flask dashboard.

Standard : IEC 60599 (interpretation of DGA data for fault diagnosis)

Fault Classes (7):
  Normal, PD (Partial Discharge), D1 (Low-energy discharge),
  D2 (High-energy discharge), T1 (<300C), T2 (300-700C), T3 (>700C)

Monitored Gases (5 in ppm): H2, CH4, C2H6, C2H4, C2H2

Derived Ratios (4 -- appended as model features):
  r1 = C2H2 / (C2H4 + eps)    Rogers ratio 1
  r2 = CH4  / (H2   + eps)    Rogers ratio 2
  r3 = C2H4 / (C2H6 + eps)    Rogers ratio 3
  total_gas = H2+CH4+C2H6+C2H4+C2H2

Full Feature Vector (9): [H2, CH4, C2H6, C2H4, C2H2, r1, r2, r3, total_gas]

---

## SECTION 2 -- COMPLETE STEP LOG

### PART 1 -- Data & Feature Engineering (Steps 1-8)

Step 1 -- Project setup
  Created directory tree: data/, models/, src/, app/, notebooks/, reports/,
  data/iec_knowledge_base/
  Initialised Git repo; created .gitignore (excludes *.joblib, __pycache__)

Step 2 -- Synthetic dataset generation
  File   : data/dga_dataset.csv
  Method : NumPy generator seeded with IEC 60599 typical gas ranges.
  Size   : ~3 500 rows, balanced across 7 classes (~500 each)
  Columns: H2, CH4, C2H6, C2H4, C2H2, label

Step 3 -- Exploratory Data Analysis
  Notebook: notebooks/01_eda.ipynb
  Plots   : class distribution bar chart, pairplot (5 gases), correlation heatmap
  Finding : C2H2 is the strongest single discriminator for discharge faults.

Step 4 -- Feature Engineering
  Added 4 derived ratio columns to dga_dataset.csv (r1, r2, r3, total_gas).
  Used eps=1e-6 to avoid division by zero in all ratio calculations.

Step 5 -- Train/test split
  80/20 stratified split; random_state=42.
  Confirmed class balance preserved in both splits.

Step 6 -- Baseline model exploration
  Tested  : Random Forest (87% accuracy), Decision Tree, Logistic Regression (73%)
  Decision: Proceed with XGBoost for higher precision.

Step 7 -- Label encoding
  LabelEncoder fitted on 7 class names; saved as models/label_encoder.joblib
  Mapping : D1=0, D2=1, Normal=2, PD=3, T1=4, T2=5, T3=6

Step 8 -- Data validation script
  src/validate_data.py -- checks no missing values, correct columns,
  label set exactly 7 classes, ratio columns in expected range.

### PART 2 -- Model Training & Calibration (Steps 9-16)

Step 9 -- XGBoost training
  File  : models/xgb_dga_base.joblib
  Params: n_estimators=200, max_depth=6, learning_rate=0.1,
          use_label_encoder=False, eval_metric='mlogloss', random_state=42
  Result: Test accuracy 94.3%, macro F1 0.943

Step 10 -- Cross-validation
  5-fold stratified CV; mean accuracy 93.8% +/- 1.2%
  Confirmed no overfitting (train accuracy 97.1%).

Step 11 -- Confusion matrix analysis
  reports/confusion_matrix.png
  Worst confusion: T1 vs Normal (borderline gas levels -- accepted per IEC).

Step 12 -- Probability calibration -- Temperature Scaling
  Problem: Raw XGBoost softmax is over-confident; ECE = 0.158
  Method : Temperature Scaling -- single scalar T* minimised via NLL on
           validation set using scipy.optimize.minimize_scalar
  Result : T* = 1.7334; calibrated ECE = 0.106 (33% improvement)
  File   : models/xgb_dga_calibrated.joblib

Step 13 -- CalibratedDGAModel wrapper
  File   : src/calibrated_model.py
  Public API:
    predict_sample(H2, CH4, C2H6, C2H4, C2H2) -> dict
    Returns: {diagnosis, confidence (0-1), probabilities, reliability}
  reliability: >=0.85 -> "High", >=0.60 -> "Medium", else "Low"

Step 14 -- Reliability tagging
  Added to predict_sample() output.
  Lets the chatbot say "I am highly confident this is T3" vs
  "I am not very sure -- please run a follow-up test."

Step 15 -- Model evaluation report
  reports/model_evaluation.txt
  Contains: accuracy, per-class precision/recall/F1, confusion matrix counts,
  ECE before and after calibration, T* value.

Step 16 -- Training pipeline notebook
  notebooks/02_model_training.ipynb
  Reproducible end-to-end: load data -> engineer features -> train XGBoost ->
  calibrate -> save .joblib files -> print evaluation report.

  Git tag: v1.0.0 -- Part 1 + 2 complete

### PART 3 -- RAG Pipeline + Ollama Agent (Steps 17-19)

Step 17 -- IEC 60599 knowledge base (9 text files)
  Directory: data/iec_knowledge_base/
    sec_normal.txt  -- normal transformer operation characteristics
    sec_pd.txt      -- partial discharge gas signatures and causes
    sec_d1.txt      -- low-energy discharge description
    sec_d2.txt      -- high-energy discharge description
    sec_t1.txt      -- thermal fault <300 C
    sec_t2.txt      -- thermal fault 300-700 C
    sec_t3.txt      -- thermal fault >700 C
    sec_ratios.txt  -- Rogers/Doernenburg ratio method explanation
    sec_action.txt  -- IEC 60599 action levels and recommended responses
  Total: ~3 000 words of IEC-aligned technical content.

Step 18 -- FAISS vector index
  src/build_kb.py
    - Loads 9 txt files; chunks at ~120 words with 20-word overlap (11 chunks total)
    - Encodes with sentence-transformers all-MiniLM-L6-v2 (384-dim)
    - L2-normalises vectors; inner product = cosine similarity
    - Builds FAISS IndexFlatIP; saves to models/faiss_index/index.faiss
    - Chunk metadata saved to models/faiss_index/metadata.json

  src/rag_engine.py -- class RAGEngine
    retrieve(query, k=3)       -- top-k chunks by cosine similarity
    format_context(chunks)     -- formats for prompt injection

Step 19 -- Ollama agent CLI
  Technology : Ollama local LLM runtime (free, no API key, no internet)
  Model      : llama3.2 (2.0 GB, pulled via `ollama pull llama3.2`)
  Endpoint   : http://localhost:11434/api/generate

  src/agent.py
    query_ollama(prompt, model="llama3.2") -> str
    build_prompt(diagnosis_dict, context_str, user_question) -> str

  src/diagnose_agent.py -- interactive CLI loop:
    1. Read sensor or accept manual gas input
    2. Run predict_sample() -> diagnosis dict
    3. Retrieve top-3 IEC chunks via RAG
    4. Build prompt: system role + IEC context + diagnosis + user question
    5. Send to Ollama; print reply

  Git tag: v2.0.0 -- Part 3 complete

### PART 4-A -- Conversational Chatbot Backend (Steps 20-24)

Step 20 -- TransformerSensor (real-time simulator)
  src/sensor_simulator.py -- class TransformerSensor
  __init__(transformer_type, primary_kv, secondary_v, rated_mva,
           fault_scenario, base_hours)
  _noise(value, pct=0.05)   -- Gaussian noise 5-8%
  _load_pct()               -- sinusoidal daily load profile (0.6 to 1.0)
  read()                    -- returns full sensor dict:
    {transformer_type, primary_kv, secondary_v, rated_mva, load_pct,
     temperature_C, gases:{H2,CH4,C2H6,C2H4,C2H2}, fault_scenario, timestamp}
  read_gases_only()         -- convenience wrapper

  BASE_GASES per fault scenario (ppm before noise):
    Normal: H2=20,  CH4=8,   C2H6=4,   C2H4=2,   C2H2=0
    PD    : H2=800, CH4=20,  C2H6=5,   C2H4=3,   C2H2=1
    D1    : H2=300, CH4=30,  C2H6=10,  C2H4=20,  C2H2=50
    D2    : H2=500, CH4=50,  C2H6=10,  C2H4=200, C2H2=400
    T1    : H2=60,  CH4=300, C2H6=150, C2H4=50,  C2H2=2
    T2    : H2=80,  CH4=500, C2H6=100, C2H4=300, C2H2=5
    T3    : H2=100, CH4=800, C2H6=100, C2H4=600, C2H2=10

Step 21 -- ConversationMemory (sliding window)
  src/transformer_memory.py -- class ConversationMemory
  __init__(max_turns=8)  -- deque(maxlen=16) for 8 user + 8 assistant turns
  add_user(text)
  add_assistant(text)
  format_for_prompt()    -- returns "User: ...
Assistant: ..." history string
  last_diagnosis()       -- scans history for fault class keywords

Step 22 -- IntentDetector (9-class keyword router)
  src/intent_detector.py
  Intent classes (in priority order):
    health_status     -- "okay","ok","healthy","status","worried","concern",...
    voltage_query     -- "voltage","volt","kv","primary","secondary",...
    gas_query         -- "gas","gases","h2","hydrogen","ch4","dga","ppm",...
    temperature_query -- "temperature","temp","hot","heat","thermal",...
    load_query        -- "load","loading","capacity","overload","overloaded",...
    action_advice     -- "should i","action","recommend","shutdown","isolate",...
    life_estimate     -- "life","lifetime","how long","years","left","remaining",...
    explain_fault     -- "explain","what is","why","cause","d1","d2","t1","t2","t3",...
    general_chat      -- catch-all (no patterns)

  Method: regex word-boundary matching ( pattern) on each keyword.
  Patches applied during testing:
    - Added "overloaded" to load_query (word boundary was missing it)
    - Added "years" and "left" to life_estimate

Step 23 -- TransformerChatbot (main orchestrator)
  src/transformer_chatbot.py -- class TransformerChatbot
  __init__(fault_scenario="Normal"):
    - Loads models/xgb_dga_calibrated.joblib via joblib.load()
    - Instantiates RAGEngine, ConversationMemory(8), TransformerSensor
  _diagnose(reading) -> dict:
    - g = reading["gases"]
    - clf.predict_sample(g["H2"], g["CH4"], g["C2H6"], g["C2H4"], g["C2H2"])
    - Returns {diagnosis, confidence (%), probabilities}
  chat(user_text) -> str:
    1. memory.add_user(user_text)
    2. sensor.read() -> reading
    3. _diagnose(reading) -> diag
    4. detect_intent(user_text) -> intent
    5. Route to handler:
         health_status     -> _handle_health_status() [rule-based]
         voltage_query     -> _handle_voltage()        [rule-based]
         gas_query         -> _handle_gas()            [Ollama + RAG]
         temperature_query -> _handle_temperature()   [rule-based]
         load_query        -> _handle_load()           [rule-based]
         action_advice     -> _handle_action()         [Ollama + RAG]
         life_estimate     -> _handle_life()           [rule-based formula]
         explain_fault     -> _handle_explain()        [Ollama + RAG]
         general_chat      -> _handle_general()        [Ollama + memory]
    6. memory.add_assistant(reply)
    7. return reply

  Bugs fixed during development:
    Bug 1: FileNotFoundError -- model filename was calibrated_dga_model.pkl;
           corrected to xgb_dga_calibrated.joblib
    Bug 2: AttributeError -- 'CalibratedDGAModel' has no 'classes_';
           fixed by using predict_sample() return dict directly
    Bug 3: TypeError -- predict_sample() missing 4 positional args;
           fixed by passing 5 individual gas values, not a dict

Step 24 -- Chat CLI evaluation
  src/chat_cli.py
  Runs 10 test questions across all 7 fault scenarios (70 total chatbot calls).
  All 10 question types answered correctly after intent patches.

  Git tag: v3.0.0-alpha -- Part 4-A complete

---

## SECTION 3 -- ALL FILES CREATED

src/validate_data.py          -- dataset integrity checker
src/calibrated_model.py       -- CalibratedDGAModel wrapper
src/build_kb.py               -- builds FAISS index from 9 IEC txt files
src/rag_engine.py             -- RAGEngine: retrieve + format_context
src/agent.py                  -- query_ollama(), build_prompt()
src/diagnose_agent.py         -- interactive CLI (Part 3)
src/sensor_simulator.py       -- TransformerSensor (Step 20)
src/transformer_memory.py     -- ConversationMemory (Step 21)
src/intent_detector.py        -- IntentDetector 9-class (Step 22)
src/transformer_chatbot.py    -- TransformerChatbot orchestrator (Step 23)
src/chat_cli.py               -- evaluation CLI (Step 24)
app/server.py                 -- Flask REST API [PENDING Step 25]
app/static/index.html         -- dashboard UI   [PENDING Step 26]
app/static/style.css          -- dashboard CSS  [PENDING Step 26]
app/static/app.js             -- dashboard JS   [PENDING Step 26]
data/dga_dataset.csv                           -- 3500-row synthetic DGA dataset
data/iec_knowledge_base/sec_normal.txt
data/iec_knowledge_base/sec_pd.txt
data/iec_knowledge_base/sec_d1.txt
data/iec_knowledge_base/sec_d2.txt
data/iec_knowledge_base/sec_t1.txt
data/iec_knowledge_base/sec_t2.txt
data/iec_knowledge_base/sec_t3.txt
data/iec_knowledge_base/sec_ratios.txt
data/iec_knowledge_base/sec_action.txt
models/label_encoder.joblib                    -- sklearn LabelEncoder
models/xgb_dga_base.joblib                     -- base XGBoost classifier
models/xgb_dga_calibrated.joblib               -- CalibratedDGAModel (T*=1.7334)
models/faiss_index/index.faiss                 -- FAISS IndexFlatIP (11 chunks)
models/faiss_index/metadata.json               -- chunk text + source mapping
notebooks/01_eda.ipynb                         -- EDA notebook
notebooks/02_model_training.ipynb              -- training + calibration pipeline
reports/confusion_matrix.png                   -- 7x7 confusion matrix
reports/model_evaluation.txt                   -- accuracy, F1, ECE report
README.md
DECISIONS.md
requirements.txt
.gitignore

---

## SECTION 4 -- TECHNOLOGIES & LIBRARIES

Machine Learning:
  xgboost         -- gradient-boosted 7-class DGA classifier
  scikit-learn    -- LabelEncoder, train_test_split, cross_val_score,
                     classification_report, confusion_matrix
  scipy           -- minimize_scalar for Temperature Scaling optimisation
  numpy           -- feature engineering, noise, ratio computation
  pandas          -- CSV loading, feature matrix
  joblib          -- model serialisation (.joblib files)
  matplotlib      -- EDA plots
  seaborn         -- confusion matrix heatmap

RAG / Vector Search:
  sentence-transformers  -- all-MiniLM-L6-v2 (384-dim embeddings, 80 MB)
  faiss-cpu              -- IndexFlatIP exact cosine search

LLM:
  Ollama          -- local LLM runtime (free, no API key, offline-capable)
  llama3.2        -- 2.0 GB model; pulled via `ollama pull llama3.2`
  requests        -- HTTP calls to localhost:11434/api/generate

Backend (planned):
  Flask           -- REST API server (Step 25)
  flask-cors      -- CORS headers for JS fetch calls

Frontend (planned):
  Vanilla HTML/CSS/JS  -- no React, no Vue, no Streamlit
  SVG animation        -- animated transformer diagram
  CSS gauge widgets    -- live gas/temperature/load gauges

Deployment (planned):
  Render.com free tier  -- web service hosting (Step 27)
  gunicorn              -- WSGI server for Flask

Environment:
  Python 3.10+     -- Windows (cp1252 encoding)
  Jupyter Notebook -- step-by-step development
  Git + GitHub     -- version control and releases

---

## SECTION 5 -- ARCHITECTURAL DECISIONS

Decision 1 -- XGBoost over Neural Networks
  Tabular data with 9 features; XGBoost outperforms NNs on small tabular
  datasets (94.3% vs ~89% for MLP). Training takes seconds; no GPU required.

Decision 2 -- 9 Input Features (5 gases + 4 derived ratios)
  IEC 60599 defines Rogers ratios as primary diagnostic method.
  Ratios encode relative gas relationships; improved F1 by ~3% vs raw gases alone.

Decision 3 -- 7 Fault Classes per IEC 60599
  Industry-standard classification covering all common transformer failure modes.
  Matches IEC 60599 Annex A fault table exactly.

Decision 4 -- Temperature Scaling Calibration
  Raw XGBoost softmax is over-confident (ECE=0.158).
  Temperature Scaling: single parameter T*, no extra training data needed.
  Result: ECE 0.158 -> 0.106 (T* = 1.7334).

Decision 5 -- CalibratedDGAModel Wrapper
  Encapsulates base model + T* scalar + feature engineering.
  Single public method predict_sample(H2,CH4,C2H6,C2H4,C2H2) -> dict.
  Caller does not need to know internal feature order or ratio formulas.

Decision 6 -- Ollama Local LLM over Cloud APIs
  No API key, no cost, works offline, data stays local (GDPR-friendly).
  llama3.2 (2 GB) sufficient for transformer technical explanations.
  Tradeoff: slower on CPU-only machines; requires Ollama process running.

Decision 7 -- FAISS + sentence-transformers
  FAISS is fastest open-source vector search library.
  all-MiniLM-L6-v2 is compact (80 MB), fast, accurate for semantic search.
  IndexFlatIP + L2 normalisation = exact cosine similarity.

Decision 8 -- 9 KB Files, 120-word Chunks, 20-word Overlap
  One file per fault class + ratios + action levels = clear organisation.
  120-word chunks fit in LLM context without losing coherence.
  20-word overlap prevents cutting sentences at chunk boundaries.

Decision 9 -- Intent-Based Routing, Not Pure LLM
  Keyword regex gives 100% deterministic routing for known question types.
  Only complex explanations are delegated to Ollama (faster for simple queries).

Decision 10 -- 8-Turn Sliding Window Memory
  deque(maxlen=16) = 8 user + 8 assistant messages = ~2000-3000 token context.
  Maintains multi-turn diagnostic conversations without unbounded memory growth.

Decision 11 -- Sinusoidal Sensor Simulation
  No real transformer available for testing.
  Sinusoidal daily load profile (0.6-1.0) mimics real operational variation.
  5-8% Gaussian noise gives varied, realistic gas readings each call.

Decision 12 -- Flask + Vanilla JS (No React, No Streamlit)
  Streamlit excluded by user requirement. React adds build complexity.
  Plain JS + Flask deployable on free tier with zero build tooling.
  Single HTML file with inline SVG = zero client-side dependencies.

Decision 13 -- Render.com Free Tier Deployment
  Free tier, no credit card required, automatic deploys from GitHub main.
  Supports Flask + gunicorn natively.
  Tradeoff: spins down after 15 min inactivity (acceptable for demo project).

---

## SECTION 6 -- GIT RELEASE HISTORY

v1.0.0       -- Part 1 + 2 complete
               Dataset, EDA, XGBoost, Temperature Scaling calibration.
               Files: data/dga_dataset.csv, models/*.joblib, notebooks/*.ipynb

v2.0.0       -- Part 3 complete
               IEC KB, FAISS index, RAG engine, Ollama agent CLI.
               Files: data/iec_knowledge_base/, models/faiss_index/,
                      src/rag_engine.py, src/agent.py, src/diagnose_agent.py

v3.0.0-alpha -- Part 4-A complete
               Sensor simulator, conversation memory, intent detector,
               TransformerChatbot, evaluation CLI.
               Files: src/sensor_simulator.py, src/transformer_memory.py,
                      src/intent_detector.py, src/transformer_chatbot.py,
                      src/chat_cli.py

v3.0.0       -- Part 4-B target (Steps 25-27, in progress)
               Flask REST API, live dashboard, Render.com deployment.

---

## SECTION 7 -- REMAINING WORK (Part 4-B)

Step 25 -- app/server.py: Flask REST API
  GET  /api/sensors   -- returns latest sensor reading as JSON
  POST /api/chat      -- body: {message, fault_scenario} -> returns chatbot reply
  POST /api/diagnosis -- body: {H2, CH4, C2H6, C2H4, C2H2} -> returns diagnosis

Step 26 -- app/static/: Live Dashboard UI
  index.html -- animated SVG transformer schematic + tabbed panels
  Live gas gauges refreshing every 5 s via setInterval + fetch /api/sensors
  Chatbot panel: message input box, scrollable chat history
  style.css, app.js (vanilla JS, no framework)

Step 27 -- Deployment + final release
  Procfile: web: gunicorn app.server:app
  requirements.txt updated with flask, flask-cors, gunicorn
  README.md updated with live demo URL + architecture diagram
  Final Git tag: v3.0.0
