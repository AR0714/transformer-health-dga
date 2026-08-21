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

## 2026-08-21 — Step 12: validation

Decided: Validate the classical code against published examples before trusting it.
Result: Duval reproduced a published worked example exactly (CH4=69, C2H4=10,
C2H2≈0 → T1), and matched the published diagnosis on 28 of 34 nameable IEC TC 10
cases (82%). The ~6 mismatches are expected (method ceiling, no "Normal" zone,
boundary cases) — not bugs. Code is trusted.
Why: "Runs without error" ≠ "correct"; 28 matches (bar was ≥5) confirms no major bug.

## Standing decisions for the next steps (recorded now, to honour later)

Decided: When splitting data for validation, use a STRATIFIED split (keep each
class's proportion in every fold).
Why: Rare classes (e.g. T2) could otherwise vanish from a fold.

Decided: Report PER-CLASS metrics (precision/recall), not just overall accuracy;
never quote an IEC score without stating how the train/IEC overlap was handled.

Decided: Classical methods first, machine learning second (Part 3).
Why: The classical baseline is the number ML must beat, and the physics features
help the model.

## Pending / open items (not yet done)

- Reword the README "why this matters" paragraph in my own words.
- Commit the newest code (`key_gas.py`, `iec_ratios.py`, `duval.py`) and the four
  report PDFs to GitHub once the machine is connected.
- Step 13: score the classical baseline (the number ML must beat) — NEXT.
