# Decision Log

Every real choice in this project is recorded here the moment it is made:
the date, what was decided, what was rejected and why, and — for the code
steps — what the code did and what result it produced. Entries are only added,
never deleted. This file is the one record that cannot be reconstructed later,
and it is the source of the best answers in an interview or viva.

How to read it: each `##` heading is a date/stage. Under it, `Decided` /
`Rejected` / `Why` capture the choice; `Did` / `Result` capture the code and
its outcome in plain language.

---

## 2026-08-14 — Part 0: project foundation (before any code)

Decided: Build a fixed folder structure first — `data/`, `notebooks/`,
`src/classical/`, `models/`, `reports/`. Rejected: dumping files together later.
Why: a clean, predictable layout stays workable and reads as competent.

Decided: Put the project under Git from day one and publish it public on GitHub.
Rejected: one big upload at the end. Why: a steady commit history is credible;
Git is also a safety net (revert to any working commit).

Decided: `.gitignore` contains `data/`. Rejected: committing the datasets.
Why: DGA datasets are large / licence-restricted — never upload to a public repo.

Decided: Add an MIT `LICENSE`. Rejected: no licence. Why: a public repo with no
licence forbids reuse and looks unfinished.

Decided: Write the README before the code, leading with the problem and its cost
(a large transformer ≈ USD 2–10 M, lead time up to ~2+ years), technique second.
Rejected: "an ML project for fault detection." Why: leading with the asset and
cost is what a recruiter reads first.

Decided: Keep two documentation registers — personal (learning journey) and
professional (what the system does / how validated) — separate.

## 2026-08-16 — Part 1: meet the data (Steps 6–8)

Decided: Use the IEEE DataPort "DGA Dataset" (DOI 10.21227/27vy-h479): three
files — DGA_train (584), DGA_test_unseen (70), IEC_TC_10 (49) — in `data/`.
Rejected: unverified Kaggle sets; the cyber-security "Domain Generation
Algorithm" datasets. Why: packages train + unseen test + IEC benchmark,
comparable to published papers.

Decided: Keep the three files separate. Why: the test set must stay a sealed exam;
IEC is an external reference. Merging destroys both roles.

Did (Step 7, load & look): `pd.read_excel` → DataFrame `df`; checked `df.shape`,
columns, `df.head()`, `df.isnull().sum()`.
Result: (584, 6) — 5 gases + `Fault class`; zero missing values (clean).

Did (Step 8, count classes): `df['Fault class'].value_counts()`.
Result: uneven — D2=134 (biggest), T2=45 (smallest): 0:85 1:59 2:76 3:134 4:96
5:45 6:89. Decided (record): lazy-baseline floor = 134/584 ≈ 23% (always guess
the majority). Any model must beat this. Fault-code map: 0 Normal, 1 PD, 2 D1,
3 D2, 4 T1, 5 T2, 6 T3.

## 2026-08-16 — Data findings that shape everything (from Document II)

Record: The training set is NOT truly "balanced" (T2=7.7% … D2=22.9%, a 3× spread).
Record: The "unseen" test set is exactly balanced (10/class) — curated, not a
real-world distribution. Only IEC TC 10 shows natural imbalance.
CRITICAL: Do NOT use IEC TC 10 as a naive benchmark after training on train —
41 of its 49 rows (84%) are duplicated inside the training file (data leakage).
Handle: benchmark only on the 8 non-overlapping rows, or de-duplicate, and always
disclose. Record: zeros are handled inconsistently (true 0 and 0.0001 placeholder);
2 exact duplicate rows in train.

Decided: Understand the data deeply before modelling (produced Document II).
Rejected: straight to modelling. Why: a model on misunderstood data is a guess;
this pause caught the IEC leakage.

## 2026-08-20 — Part 2: physics into code (Steps 9–13, classical methods)

Did (Step 9): `src/classical/key_gas.py` — dominant-gas → fault. Simplified to the
5 available gases (no CO). Note: crudest method, a starting point only.

Did (Step 10): `src/classical/iec_ratios.py` — the three IEC 60599 ratios with safe
division (None on divide-by-zero), a low-gas guard, and an explicit "No decision".
Why: unlike Key Gas it may honestly refuse (row 0 → No decision).

Did (Step 11): `src/classical/duval.py` — Duval Triangle 1: percentages → (x,y) point
→ zone, with a plot, "DT" fallback. Trap: the triangle can NEVER say "healthy" —
it always names a fault, so abnormality must be established first.

Record: the three methods disagree on row 0 (Key Gas=PD, IEC=No decision, Duval=D1;
true=PD). Expected — the reason to cross-check and later add ML.
Lesson: `.ipynb` ≠ `.py`; after editing an imported `.py`, restart the kernel.

Did (Step 12, validation): matched a published worked example (CH4=69,C2H4=10,
C2H2≈0 → T1 ✓) and 28/34 nameable IEC TC 10 cases (82%). The ~6 misses are
expected (method ceiling, no "Normal" zone, boundary cases). Code trusted.

Did (Step 13, baseline): scored Duval on the sealed unseen test set.
Result: classical (Duval) baseline = 57.1% (40/70), vs 14.3% (test) / 22.9% (train)
lazy floor; fault-only ≈ 66.7%. THIS 57.1% is the number ML must beat.
Calibration: classical DGA at 50–70% is normal — do not chase 90%+.
>>> PART 2 COMPLETE.

## 2026-08-21 — Part 3: machine learning (Steps 14–18)

Did (Step 14, features): `build_features(df)` — 5 raw gases + 3 IEC ratios
(C2H2/C2H4, CH4/H2, C2H4/C2H6) + `total_gas` = 9 features. Used `eps = 1e-6` so
ratios never divide by zero (ML needs a finite number in every cell, not the
classical "No decision"). Result: X = (584, 9), y = (584,).
Why: hand the model the physics ratios engineers use — a head start on small data.

Did (Step 15, split): `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
Result: 467 train / 117 validation; class % preserved across parts (T2 = 7.7% in
full, train and val). Why: stratify so rare classes don't vanish; `random_state`
makes the split reproducible. The separate unseen test file stays sealed.

Did (Step 16, first model): `RandomForestClassifier(n_estimators=300)`, trained on
the training part, checked on validation. Result: validation accuracy = 81.2%
(beats the 57.1% baseline). Why: Random Forest is the right first model for small
tabular data.

Did (Step 17, cross-validate): 5-fold `StratifiedKFold`.
Result: fold scores [80.3, 82.1, 86.3, 83.8, 77.6]; mean = 82.0% ± 3.0%. Confirms
the 81% was real and stable, not a lucky split. Decided: always report mean ± std,
never the single best fold.

Did (Step 18, second model + final test): installed xgboost 3.4.1; trained Random
Forest and XGBoost on the FULL training data; tested on the SEALED unseen test set
(opened only now, for a fair final score).
FINAL RESULT (unseen test):
    Classical (Duval) baseline : 57.1%
    Random Forest              : 78.6%
    XGBoost                    : 80.0%
ML beats the classical baseline by ~23 points on unseen data.
Decided: XGBoost is the final model (recognised name, slight edge). Note it is
statistically TIED with Random Forest (1.4 pts on a 70-row test = within noise),
so Random Forest would be an equally valid, simpler choice.
Note: substituted the handbook's XGBoost with a one-time `pip install xgboost`;
scikit-learn's HistGradientBoosting was the no-install equivalent used meanwhile.
Per-class (XGBoost, unseen test — honest view, not just the 80% headline):
strong T1/T2/T3 (f1 0.91–0.95); D2 arcing recall 0.80 (catches 8/10, MISSES 2 —
a false-negative worth flagging); D1 weakest (precision 0.55); Normal recall 0.60
(over-flags some healthy units — the safer error). Report per-class, always.
Calibration: 80% on a small, imbalanced set is a solid, honest result — NOT a
claim of deployment-grade field accuracy (Document II limitations still hold).
>>> PART 3 COMPLETE.

## 2026-08-21 — Part 4: make it trustworthy (Steps 19–21)

Did (Step 19, calibrate): wrapped XGBoost in `CalibratedClassifierCV(method='isotonic',
cv=5)`, fit on the full training set (X, y), measured on the SEALED unseen test. Built a
calibration curve (predicted confidence vs actual accuracy) + Expected Calibration Error
(ECE, one honesty number, lower = better). Saved `reports/calibration_curve.png`.
Result: raw XGBoost is OVERCONFIDENT — mean confidence 0.94 vs true accuracy 0.80,
ECE ≈ 0.15. After isotonic calibration mean confidence drops to ~0.85 and ECE roughly
halves to ~0.10, with accuracy UNCHANGED (calibration re-scales confidence, not the
winning class). Checked sigmoid too — isotonic won, matching the handbook.
Why it matters: the system ranks transformers by risk; dishonest probabilities make a
dishonest priority list. Calibrate BEFORE the risk ranking (Step 22).
Read the curve: a point below-right of the diagonal = overconfident there;
above-left = underconfident. 70-row test → sparse bins/jagged curve; the direction
(overconfident → corrected) is robust, exact per-bin values are noisy.

Did (Step 20, explainability): `shap.TreeExplainer(xgb)` on the unseen test → 3-D
SHAP array (rows, features, classes). Saved `reports/shap_summary.png` (overall
importance by class) and `reports/shap_waterfall.png` (one prediction explained).
Note: handbook's `shap_values[0]` is the OLD API; modern SHAP (0.51) returns a 3-D
array, so slice `sv.values[:,:,class]` explicitly.
Result — the model learned real physics (a validation, not just a picture): top feature
is C2H2 (acetylene), its bar dominated by D2 = the arcing signature; next three are the
engineered IEC ratios (Step 14 features earning their keep). Row 0 (true PD, pred PD):
low CH4/H2 ratio (0.07) and high H2 (2240 ppm) drove it to PD — textbook partial-discharge
chemistry. "Done when" answer: H2 at 2240 ppm drove row 0 to partial discharge.

Did (Step 21, confusion matrix): `ConfusionMatrixDisplay.from_estimator(xgb, X_test,
y_test, display_labels=names)` → `reports/confusion_matrix.png`. Diagonal 56/70 = 80%.
Result: thermal faults near-perfect (T1 10/10, T3 10/10, T2 9/10). ALL errors cluster in
the discharge family — PD↔D1↔D2 (top: PD→D1 3×, then D1↔D2 / D1↔PD 2× each). Physically
sensible: PD/D1/D2 are one discharge phenomenon at rising energy, so mixing neighbours is
expected. NO absurd cross-family error (no discharge called thermal) = strong evidence the
model learned real DGA physics. Weakest class = Normal (6/10 recall; 4 healthy flagged as
mild faults) — the SAFE error direction, disclosed.
>>> PART 4 COMPLETE (model is calibrated, explainable, diagnostically sound).

## 2026-08-21 — Part 5: make it a product (Steps 22–25)

Did (Step 22, fleet risk ranking): risk = expected severity = Σ (calibrated P(fault) ×
severity weight). Weights from IEC 60599 fault energy: Normal 0.0, PD 0.2, T1 0.3, T2 0.5,
D1 0.6, T3 0.8, D2 1.0. Used the CALIBRATED probs (Step 19) so the ranking is honest.
Saved `reports/fleet_risk_ranking.csv`, sorted worst-first.
Result (validation): top 8 units all truly D2 arcing (risk ~1.0), then T3 (~0.8); 19 of
top 20 are truly serious (D1/D2/T3); true-Normal units sink to mean rank ~58/70. One false
alarm in the top 20 (TX-066 Normal→T3) — safe-direction error, disclosed.
Decided: keep the formula deliberately simple (prob × severity, one line) — defensible beats
elaborate. No `trend` term: data is single snapshots, not time series (stated, not faked).

Did (Step 23, dashboard): built a single-file, self-contained HTML fleet-health dashboard
(`reports/dashboard.html`), styled minimalist-editorial with a dark/light toggle, powered by
the REAL calibrated model output for all 70 unseen-test units.
Planned: mirror a SaaS "customer health" dashboard template (Lovable) but for transformers.
Decided: build it as a self-contained HTML file (with Claude Code), NOT on Lovable — a file
that lives in the repo is portable, deployable (GitHub Pages), portfolio-embeddable, and
unambiguously mine. Rejected: building inside Lovable's hosted platform (ownership/portability).
Health score = 100 − risk (0–100 scale). KPIs: units at risk (36/70), median health (48),
flagged arcing/D2 (10), mean risk (0.52). Panels: KPI row; a risk-distribution beeswarm
(each unit a circle, x = health, colour by band green/amber/red, threshold line at 50,
segment tabs Discharge/Thermal/Normal); a worst-first watchlist (fault chip, health bar,
risk, model confidence, dominant gas).
Result: the D2 arcing units cluster at the critical end and healthy units at the safe end —
the model's physics made visible; a real operator triage view.
Honest notes: these 70 units are the curated evaluation set (deliberately fault-heavy), NOT a
real utility fleet (which skews mostly healthy); NO trend lines / renewal dates because the
data is single snapshots — those need weekly history (a future hardware-streaming feature).
Colour: status palette validated colour-blind-safe, and every band also shows the numeric
health, so colour is never the only signal.
>>> Step 23 COMPLETE. Remaining in Part 5: Step 24 finish README, Step 25 wrap-up.

Note: career/visibility work (résumé v1.1 update, LinkedIn launch plan, portfolio, paper
positioning) is tracked in the separate "Industrial AI Sprint" chat, not here — this log
stays project/engineering only.

## Files / code created so far (artifacts)

- `src/classical/key_gas.py`    — Key Gas method (dominant gas → fault).
- `src/classical/iec_ratios.py` — IEC 60599 three-ratio method + "No decision".
- `src/classical/duval.py`      — Duval Triangle 1: percentages, coords, zone, plot.
- `notebooks/01_load_and_look.ipynb` — load/inspect data; classical scoring;
  `build_features`; split; Random Forest; cross-validation; XGBoost; calibration;
  SHAP; confusion matrix; fleet risk ranking.
- `reports/calibration_curve.png` — Step 19 (raw vs isotonic-calibrated).
- `reports/shap_summary.png`, `reports/shap_waterfall.png` — Step 20 explainability.
- `reports/confusion_matrix.png` — Step 21.
- `reports/fleet_risk_ranking.csv` — Step 22 ranked fleet, worst-first.
- `reports/dashboard.html` — Step 23 self-contained fleet-health dashboard.
- `reports/` also holds four documentation PDFs (Foundation, Dataset Forensics,
  Progress Log, Project Journey); this decision log lives at the repo root.
- Structure (from Step 2): data/ notebooks/ src/classical/ models/ reports/.
- To do later: move `build_features` into `src/features.py`; save the final model
  into `models/`.

## Standing decisions carried forward

Stratified splits always (rare classes must keep their proportion).
Report PER-CLASS metrics (precision/recall/F1), not just overall accuracy.
Every model is judged against the classical baseline (57.1%); the final ML model
is judged on the SEALED unseen test set only.
No deep learning at 584 rows — tree models (Random Forest / XGBoost) instead.
Calibrated probabilities (not raw model confidence) feed any risk ranking.
Build shippable artifacts as self-contained files in the repo (own it), not on hosted
no-code platforms.

## Pending / open items

- Reword the README "why this matters" paragraph in my own words.
- Save `reports/dashboard.html` into the repo; verify the GitHub repo link works.
- Commit + push the notebook, all `reports/` figures/CSV/HTML, and this log.
- Part 4 DONE (Steps 19–21). Part 5: Steps 22 (risk ranking) + 23 (dashboard) DONE.
  Step 24 finish README — NEXT, then Step 25 wrap-up.
- Later extensions: hardware sensing prototype; IEEE paper (see Sprint chat).
