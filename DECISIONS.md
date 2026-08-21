# Decision Log

Every real choice in this project gets recorded here the moment it is made:
the date, what was decided, and what was rejected and why. Entries are only
added, never deleted. This file is the one record that cannot be reconstructed
later — it is the source of the best answers in an interview or viva.

Format: `## DATE` then `Decided:` / `Rejected:` / `Why:` lines.

---

## 2026-08-14  — Project foundation (Part 0)

Decided: Build a fixed folder structure before writing any code —
`data/`, `notebooks/`, `src/classical/`, `models/`, `reports/`.
Rejected: Dumping all files together and organising "later".
Why: A clean, predictable layout stays workable at week eight and reads as
competent to a reviewer; ad-hoc folders turn into forty files named final_v2.

Decided: Put the project under Git from day one and publish it public on GitHub.
Rejected: A single upload at the end of the project.
Why: A commit history showing steady work over weeks is credible; one final-day
upload is not. Git also gives a safety net (revert to any working commit).

Decided: Add a `.gitignore` containing `data/` (plus Python/OS cruft).
Rejected: Committing the datasets to the repo.
Why: DGA datasets are large and often licence-restricted; they must not be
uploaded to a public repo. Data lives locally only.

Decided: Add an MIT `LICENSE`.
Rejected: No licence ("None").
Why: A public portfolio repo with no licence legally forbids reuse and looks
unfinished; MIT is the standard permissive choice.

Decided: Write the README before the code, leading with the problem and its
cost (large transformer ≈ USD 2–10 M, lead time up to ~2+ years), technique second.
Rejected: Describing it as "an ML project for fault detection" and stopping.
Why: README-driven development forces actually knowing the problem; leading with
the asset and cost is what a recruiter reads first.

Decided: Keep two documentation registers — a personal (learning journey) and a
professional (what the system does / how validated) — kept separate.
Why: Interview answers and honest limitations live in different registers.

## 2026-08-16 — Dataset choice (Part 1, Step 6)

Decided: Use the IEEE DataPort "DGA Dataset" (DOI 10.21227/27vy-h479) as the
primary dataset: three files — DGA_train (584), DGA_test_unseen (70),
IEC_TC_10 (49) — placed in `data/`.
Rejected: Unverified Kaggle "DGA" sets; and the cyber-security "Domain
Generation Algorithm" datasets that share the letters DGA.
Why: The IEEE set packages a train set, an unseen test set, and the
internationally recognised IEC TC 10 benchmark, making results comparable to
published papers. Confirmed correct by checking for the five gases + fault label.

Decided: Keep the three files separate (do not merge).
Rejected: Combining them into one file.
Why: The test set must stay an unseen exam; the IEC file is an external reference.
Merging would destroy both roles.

## 2026-08-16 — Data findings that shape everything downstream

Decided (record): The training set is NOT truly "balanced" as the documentation
claims — classes range from T2 = 45 (7.7%) to D2 = 134 (22.9%), a 3× spread.
Why it matters: report the discrepancy honestly rather than trusting the docs.

Decided (record): The "unseen" test set is exactly balanced (10 per class), so it
is curated, not a real-world frequency distribution. Only the IEC TC 10 file
shows natural imbalance.

Decided (CRITICAL): Do NOT use the IEC TC 10 file as a naive independent
benchmark after training on the training set.
Why: 41 of its 49 rows (84%) are duplicated inside the training file (verified by
row-by-row comparison). Using it as a post-training "benchmark" would be data
leakage and the score would be meaningless. Handling: benchmark only on the 8
non-overlapping rows, or remove the overlap first, and always disclose it.

Decided (record): Zero-handling in the data is inconsistent — some cells are true
0, others use a 0.0001 placeholder, in the same columns. Ratio code must guard
against both. Also: 2 exact duplicate rows exist in the training file.

## 2026-08-16 — Step 8: the baseline floor

Decided (record): Lazy-baseline accuracy = 134/584 ≈ 23% (always predict the
majority class, D2). Any real model must beat 23% to have learned anything.
Why: This is the yardstick every future model is measured against; it also means
plain accuracy can mislead, the data split must be stratified, and deep learning
is the wrong tool at 584 rows.

Decided: Obtained and recorded the official fault-code mapping from the IEEE page:
0 Normal, 1 PD, 2 D1, 3 D2, 4 T1, 5 T2, 6 T3.

## 2026-08-16 — Understand the data before modelling

Decided: Pause and understand the datasets deeply before any modelling; produced
a full forensic dataset document (Documentation II).
Rejected: Going straight from "get data" to "train a model".
Why: A model built on misunderstood data is a guess dressed up as a result. This
pause is what surfaced the IEC leakage above.

## 2026-08-20 — Part 2: physics into code (classical methods)

Decided (Step 9): Implement the Key Gas method as `src/classical/key_gas.py` —
one function that returns the fault of the dominant gas. Simplified to the five
available gases; dropped the CO / paper mapping.
Why: Our dataset has no CO. Key Gas is the crudest method, kept as a starting
point and a baseline component, not a final answer.

Decided (Step 10): Implement the IEC 60599 three-ratio method as
`src/classical/iec_ratios.py` — safe division (returns None on divide-by-zero),
a low-gas guard, and an explicit "No decision" output.
Why: Unlike Key Gas, this method is allowed to refuse to answer; an honest
"No decision" is preferred over a false diagnosis (row 0 → No decision).

Decided (Step 11): Implement the Duval Triangle 1 as `src/classical/duval.py` —
percentages → (x,y) point → zone, with a plot, and "DT" as the fallback zone.
Why: The centrepiece visual classical method. Trap recorded: the Duval Triangle
can NEVER output "healthy" — it always names a fault, so abnormality must be
established first (via gas limits) before trusting the zone.

Decided (record): The three methods disagree on row 0 (Key Gas = PD, IEC =
No decision, Duval = D1; true label = PD). This is expected and is the reason to
cross-check methods and, later, add ML.

Recorded (lesson): `.ipynb` (notebook) and `.py` (module) are different formats;
after editing an imported `.py` you must restart the kernel for the notebook to
see the change. (Both cost real debugging time in Steps 9–11.)

## 2026-08-21 — Step 12: validation against published examples

Decided: Validate the classical code against published examples before trusting it.
Result: Duval reproduced a published worked example exactly (CH4=69, C2H4=10,
C2H2≈0 → T1), and matched the published diagnosis on 28 of 34 nameable IEC TC 10
cases (82%). The ~6 mismatches are expected (method ceiling, no "Normal" zone,
boundary cases) — not bugs. Code is trusted.
Why: "Runs without error" ≠ "correct"; 28 matches (bar was ≥5) confirms no major bug.

## 2026-08-21 — Step 13: classical baseline scored  ← PART 2 COMPLETE

Decided: Use the Duval Triangle as the classical baseline classifier, and score it
on the UNSEEN test set (the same file the ML will be judged on).
Rejected: Scoring on the training set only, or using Key Gas (its coarse labels
like "T1/T2" don't map to a single class).
Why: Duval always returns a named fault and maps cleanly to the seven labels;
scoring on the unseen test set gives a fair, apples-to-apples baseline for ML.

Result (record): Classical (Duval) baseline = 57.1% on the unseen test set (40/70),
versus a lazy floor of 14.3% (test) / 22.9% (train). Duval on fault-only rows
(excluding the 10 "Normal" cases it structurally cannot predict) ≈ 66.7% (40/60).
**This 57.1% is the number the machine learning must beat in Part 3.**
Calibration: classical DGA scoring 50–70% is normal and respectable — do not
expect 90%+, and do not "improve" it by bending the standard to fit the data.

## Part 2 artifacts (files / code created and committed)

- `src/classical/key_gas.py`   — Key Gas method (dominant-gas → fault).
- `src/classical/iec_ratios.py`— IEC 60599 three-ratio method + "No decision".
- `src/classical/duval.py`     — Duval Triangle 1: percentages, coords, zone, plot.
- `notebooks/01_load_and_look.ipynb` — load/inspect data; run + score all methods.
- `reports/` — four documentation PDFs (Foundation, Dataset Forensics, Progress
  Log Part 0-1, Project Journey) + this decision log at the repo root.
- Structure unchanged from Step 2: data/ notebooks/ src/classical/ models/ reports/.

## Standing decisions for the next steps (Part 3 — machine learning)

Decided: When splitting data for validation, use a STRATIFIED split (keep each
class's proportion in every fold).
Why: Rare classes (e.g. T2 = 45) could otherwise vanish from a fold.

Decided: Report PER-CLASS metrics (precision/recall), not just overall accuracy;
never quote an IEC score without stating how the train/IEC overlap was handled.

Decided: Every ML model is judged against the classical baseline (57.1%). A model
that cannot beat it is not earning its place.

Decided: Classical first (done), machine learning second; use tree-based models
(Random Forest / XGBoost) or SVM, NOT deep learning, given only 584 small rows.

## Pending / open items (not yet done)

- Reword the README "why this matters" paragraph in my own words.
- Commit + push the latest notebook and this updated decision log to GitHub.
- Part 3 (Machine Learning): Step 14 (engineer features) — NEXT.
