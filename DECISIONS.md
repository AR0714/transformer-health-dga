# Architectural Decisions — Intelligent Transformer Health Monitoring Using DGA

_Last updated: Part 4 Phase A complete_

---

## PART 1 — ML Model Selection

### Decision 1: XGBoost over Neural Networks
**Context:** 7-class DGA fault classification with ~400 training samples.
**Decision:** XGBoost (gradient boosted trees) with calibration.
**Reason:** Neural networks overfit on small tabular datasets. XGBoost gives
interpretable feature importances, trains in seconds, and achieves 97%+ accuracy
on DGA gas ratio features without a GPU.
**Rejected:** MLP, LSTM, SVM.

### Decision 2: 9 Input Features (5 raw gases + 4 ratios)
**Context:** IEC 60599 uses gas ratios (Rogers, Doernenburg) for fault diagnosis.
**Decision:** Feature vector = [H2, CH4, C2H6, C2H4, C2H2, C2H2/(C2H4+eps),
CH4/(H2+eps), C2H4/(C2H6+eps), total_gas].
**Reason:** Raw gas ppm alone gives ambiguous diagnoses across fault types.
Adding the 4 key IEC ratios significantly improves class separation.
**Outcome:** Accuracy improved from 89% (raw only) to 97%+ (raw + ratios).

### Decision 3: 7 Fault Classes
**Context:** IEC 60599 defines fault categories for DGA.
**Decision:** Normal, PD, D1, D2, T1, T2, T3 — exactly as per IEC 60599 Table 1.
**Reason:** Industry-standard classification. Enables direct comparison with
manual IEC diagnosis. Recognisable to any power engineer.

---

## PART 2 — Model Calibration

### Decision 4: Temperature Scaling Calibration
**Context:** Raw XGBoost probabilities were overconfident (ECE = 0.158).
**Decision:** Post-hoc Temperature Scaling with T* = 1.7334.
**Reason:** Calibration is critical for a safety system — a 99% confident wrong
diagnosis is dangerous. Temperature Scaling is parameter-efficient (1 scalar),
provably non-distorting to accuracy, and reduces ECE from 0.158 to 0.106.
**Rejected:** Platt Scaling (less theoretically motivated for multi-class),
Isotonic Regression (overfits on small validation sets).

### Decision 5: CalibratedDGAModel Wrapper Class
**Context:** Need a single object that encapsulates the full inference pipeline.
**Decision:** Custom Python class saved with joblib as xgb_dga_calibrated.joblib.
**Reason:** Hides calibration complexity from downstream consumers. The chatbot
and Flask API call predict_sample(H2, CH4, C2H6, C2H4, C2H2) and get back
{"diagnosis", "confidence", "probabilities", "reliability"} directly.

---

## PART 3 — LLM-Powered RAG Agent

### Decision 6: Ollama (Local LLM) over Cloud APIs
**Context:** Need natural language generation for engineering reports.
**Decision:** Ollama + llama3.2 (2.0 GB), running locally on localhost:11434.
**Reason:** Zero cost, no API key, no data privacy concerns, works offline.
A final-year student project must be reproducible by anyone with a laptop.
**Rejected:** OpenAI GPT-4 (cost, API key required), Hugging Face Inference API
(rate limits, latency).

### Decision 7: FAISS + sentence-transformers for RAG
**Context:** Need semantic search over IEC 60599 knowledge base.
**Decision:** all-MiniLM-L6-v2 (384-dim) embeddings + FAISS IndexFlatIP
(cosine similarity via inner product on L2-normalised vectors).
**Reason:** all-MiniLM-L6-v2 is fast, small (80 MB), and excellent for
short technical text. FAISS IndexFlatIP gives exact cosine search with
no approximation error on a small (11-chunk) index.
**Outcome:** 11 vectors indexed, top-3 retrieval consistently finds the correct
IEC fault section with scores > 0.7.

### Decision 8: 9 IEC Knowledge-Base Files, 120-Word Chunks
**Context:** IEC 60599 has sections for each fault type plus action levels.
**Decision:** 9 text files (sec_normal.txt ... sec_action_levels.txt),
chunked at 120 words with 20-word overlap.
**Reason:** 120 words fits comfortably in the LLM context window as retrieved
context. 20-word overlap prevents important boundary information being split.
11 total chunks — small enough for exact search, large enough to cover the
standard meaningfully.

---

## PART 4 — Conversational Chatbot Backend

### Decision 9: Intent-Based Routing (not pure LLM)
**Context:** Need to answer 9 types of transformer questions reliably and fast.
**Decision:** Keyword-based intent detector routes to specialised handlers;
only health_status, gas_query, action_advice, explain_fault, and general_chat
invoke Ollama. Voltage, temperature, load queries use rule-based instant replies.
**Reason:** LLM calls take 5-30 seconds. Routing simple factual queries
(voltage = sensor reading, load = percentage) to instant rule-based handlers
gives sub-second responses for 4/9 intent types while keeping LLM quality
for complex questions.

### Decision 10: 8-Turn Sliding Window Memory
**Context:** Conversational chatbot needs context of prior exchanges.
**Decision:** ConversationMemory with deque(maxlen=16) — stores 8 complete
user+assistant turn pairs.
**Reason:** 8 turns is enough to maintain conversation coherence without
bloating the LLM prompt. Beyond 8 turns, older context is usually irrelevant
to the current question.

### Decision 11: Real-Time Sensor Simulation (not static gas values)
**Context:** Dashboard needs live-updating readings; chatbot needs current context.
**Decision:** TransformerSensor generates readings with sinusoidal daily load
profile + 5-8% Gaussian noise. Base gas levels tuned per IEC fault scenario.
**Reason:** A static demo looks unconvincing to recruiters. Animated live gauges
that actually change every few seconds demonstrate real-world applicability.
Temperature, load, and voltage all update realistically with time-of-day.

---

## PART 4 Phase B — Dashboard (Pending)

### Decision 12: Flask + Vanilla JS (no React/Streamlit)
**Context:** Need a deployable web dashboard with live gauges and chatbot.
**Decision:** Flask REST API (3 endpoints) + single HTML/CSS/JS page.
**Reason:** No build toolchain, no npm, no webpack. Flask serves the
static files and the /api/* endpoints. The whole dashboard is one HTML file
that any browser can open. Deployable to Render.com free tier in minutes.
**Rejected:** Streamlit (user preference), React (overkill, requires build step),
Django (too heavy for a demo dashboard).

### Decision 13: Render.com for Deployment
**Context:** Need a public URL that recruiters can visit.
**Decision:** Render.com free tier (Python web service).
**Reason:** Free, permanent URL, GitHub auto-deploy on push, no credit card
required. One-click deploy from GitHub. The free tier sleeps after 15 minutes
of inactivity but wakes in ~30 seconds — acceptable for a portfolio demo.
**Rejected:** Heroku (no free tier since 2022), Vercel (Python backend limited),
AWS/GCP (complex for a student demo).
