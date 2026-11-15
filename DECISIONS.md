# DECISIONS.md -- Intelligent Transformer Health Monitoring Using DGA

Author : Ankit Raj, Final-Year EE, KIIT University
Repo   : https://github.com/AR0714/transformer-health-dga

Every entry below is tied to an executed output in `notebooks/01_load_and_look.ipynb`
or to code in `src/`. Decisions that could not be tied to either were removed.
The web dashboard lives in the separate `gridpulse-dashboard` repository.

---

## 1. Overview

Goal : classify transformer faults from dissolved gas analysis (DGA) with a
       calibrated XGBoost model, explain the result, and answer questions about it
       with a local RAG + Ollama agent grounded in IEC 60599.

Fault classes (7), as numbered in the notebook (cells 15-16, 27):
  0 Normal, 1 PD, 2 D1, 3 D2, 4 T1, 5 T2, 6 T3

Gases (ppm): H2, CH4, C2H6, C2H4, C2H2

Features (9), from `build_features` (notebook cell 18; saved model confirms
the same 9 names):
  5 raw gases, r_c2h2_c2h4, r_ch4_h2, r_c2h4_c2h6 (ratios, eps = 1e-6), total_gas

---

## 2. Data and evaluation protocol

- Training file DGA_train.xlsx: 584 rows; 582 after dropping 2 exact duplicates
  (notebook cell 38).
- Sealed test file DGA_test_unseen.xlsx: 70 rows, 10 per class.
- IEC TC10 benchmark: 49 rows (used only for the leakage audit).
- The xlsx data files are not in the repository (`data/` is git-ignored).
- Lazy floor (always guess the commonest class) on the test set: 14.3% (cell 17).
- Classical Duval Triangle on the same 70 rows: 57.1% (40/70).

## 3. Models and results (all on the sealed 70-row test set unless stated)

| Model / check                                   | Result                          | Notebook cell |
|-------------------------------------------------|---------------------------------|---------------|
| Random Forest, 300 trees, validation split      | 81.2%                           | 22            |
| Random Forest, 5-fold stratified CV (train)     | 82.0% +/- 3.0%                  | 23            |
| Random Forest, test                             | 78.6%                           | 26            |
| XGBoost, n_estimators=300, test                 | 80.0%                           | 26            |
| XGBoost macro-F1, test                          | 0.80                            | 27            |
| XGBoost 95% bootstrap CI (1000 resamples)       | [70.0%, 88.6%]                  | 46            |

XGBoost hyperparameters (saved model `models/xgb_dga_base.joblib`):
n_estimators=300, random_state=42, eval_metric='mlogloss'; max_depth (6) and
learning_rate (0.3) are the XGBoost defaults.

Per-class F1 (cell 27): Normal 0.75, PD 0.70, D1 0.57, D2 0.80, T1 0.91,
T2 0.95, T3 0.91. D1 is the weakest class.

T2 has 45 rows in the raw training data, the smallest class (cell 4).

## 4. Calibration

- Final method: temperature scaling via `CalibratedDGAModel(method='temperature')`
  in `src/calibration.py` (single scalar T, found with `scipy.optimize.minimize_scalar`).
  Result: T* = 1.7334; ECE 0.1575 -> 0.1060 (cell 34).
- Caveat: T was fitted on the same 70 test rows that ECE is measured on, so the
  improvement is in-sample.
- Saved as `models/xgb_dga_calibrated.joblib`.
- Exploratory alternative (cell 28): isotonic calibration, 5-fold on the
  training data, gave ECE 0.098 on the test rows. It was not the saved model.

## 5. Abstention and reliability

- Abstention threshold: calibrated confidence below 0.50 -> "engineer review"
  (cell 36). On the test set: 66/70 answered (94.3% coverage), 4/70 abstained (5.7%).
- Reliability band in `predict_sample()` (`src/calibration.py`):
  HIGH >= 0.75, MEDIUM >= 0.50, otherwise LOW.

## 6. Leakage audit (cells 38-40)

- 42 of 49 IEC TC10 rows overlap with the deduplicated training data; 6 are unseen
  (one row is unaccounted for; a repeated IEC row is suspected, and the data files
  are not available to check).
- Accuracy with the deduplicated training set: sealed test 82.9% (original model
  80.0%); IEC TC10 all 49 rows 91.8% (inflated by leakage); the 6 unseen rows 66.7%
  (n=6, too small for strong claims).

## 7. RAG and agent (`src/`)

- Knowledge base: 9 text files in `data/iec_knowledge_base/` (sec_normal, sec_pd,
  sec_d1, sec_d2, sec_t1, sec_t2, sec_t3, sec_ratios, sec_action_levels).
- `build_kb.py`: 120-word chunks, 20-word overlap, 11 chunks in total;
  sentence-transformers all-MiniLM-L6-v2 (384-dim, normalised); FAISS IndexFlatIP
  (notebook cell 58 output).
- `rag_engine.py`: retrieval over that index.
- `agent.py`: Ollama at http://localhost:11434/api/generate, default model llama3.2.
- `diagnose_agent.py`: CLI agent (gas input -> model -> retrieval -> LLM answer).

## 8. Conversational backend (`src/`)

- `sensor_simulator.py`: `TransformerSensor` with a fixed base gas profile per fault
  scenario plus noise.
- `transformer_memory.py`: `ConversationMemory`, default 8 turns (deque of 16 messages).
- `intent_detector.py`: 9 keyword-regex intents (health_status, voltage_query,
  gas_query, temperature_query, load_query, action_advice, life_estimate,
  explain_fault, general_chat).
- `transformer_chatbot.py`: loads `models/xgb_dga_calibrated.joblib`, routes by
  intent to rule-based or Ollama + RAG handlers.
- `chat_cli.py`: CLI chatbot.

## 9. Files in the repository

```
src/ : agent.py, build_kb.py, calibration.py, chat_cli.py, classical/, classical_methods.py,
       diagnose_agent.py, intent_detector.py, predict.py, rag_engine.py,
       sensor_simulator.py, transformer_chatbot.py, transformer_memory.py
models/ : xgb_dga_base.joblib, xgb_dga_calibrated.joblib, faiss_index/
notebooks/ : 01_load_and_look.ipynb
reports/ : calibration_curve.png, classical_comparison.png, confusion_matrix.png,
           fleet_risk_ranking.csv, risk_coverage_curve.png, shap_per_class.png,
           shap_summary.png, shap_waterfall.png, MODEL_CARD.md,
           Transformer_DGA_Dataset_Understanding.pdf
```

## 10. Releases

Tags on the remote: v1.0.0, v2.0.0, v3.0.0-alpha.

## 11. Not in this repository

- The web dashboard (see `gridpulse-dashboard`).
- The raw xlsx datasets.
- Automated tests, a CI workflow and a one-command run script.
