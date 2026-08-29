"""
Doernenburg Ratio Method for DGA Transformer Fault Diagnosis
=============================================================
Reference : Doernenburg, E. & Strittmatter, W. (1974).
            Monitoring oil-cooled transformers by gas analysis.
            Brown Boveri Review, 61(5), 238-247.
            IEEE C57.104-2019, Annex C.
            IEC 60599:2022, Annex B.

Method summary
--------------
The Doernenburg method has two stages:

  Stage 1 — Minimum concentration check
      At least ONE of the four sentinel gases must exceed its L1 threshold.
      If none does, the transformer is considered in normal aging and
      the Doernenburg ratios cannot be applied (method returns N/A).

      L1 thresholds (ppm):
          H2 > 100  |  C2H2 > 35  |  CH4 > 120  |  C2H4 > 50

  Stage 2 — Four-ratio diagnosis
      Four ratios are computed and each is classified as "High" or "Low"
      relative to its critical value.  A fault type is confirmed when
      AT LEAST THREE of the four ratios satisfy the fault's criteria.

      Ratios and critical values:
          R1 = CH4 / H2       critical value = 1.0
          R2 = C2H2 / C2H4   critical value = 0.75
          R3 = C2H2 / CH4    critical value = 0.3
          R4 = C2H6 / C2H2   critical value = 0.4

      Fault criteria (at least 3 of 4 must be satisfied):
          Thermal (T)  : R1 ≥ 1.0,  R2 < 0.75, R3 < 0.3,  R4 ≥ 0.4
          Arcing  (D)  : R1 < 1.0,  R2 ≥ 0.75, R3 ≥ 0.3,  R4 < 0.4
          Corona  (PD) : R1 < 0.1,  R2 < 0.75, R3 < 0.3,  R4 ≥ 0.4

      If no fault type scores 3+, result is "Undetermined".
      If two types tie on score, the one with more matching ratios wins.
"""

__all__ = ["doernenburg"]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Stage-1 minimum concentrations (ppm) — at least one must be exceeded
_L1 = {"H2": 100.0, "C2H2": 35.0, "CH4": 120.0, "C2H4": 50.0}

# Stage-2 ratio critical values
_CRITICAL = {
    "R1_CH4_H2":    1.00,   # CH4/H2
    "R2_C2H2_C2H4": 0.75,  # C2H2/C2H4
    "R3_C2H2_CH4":  0.30,  # C2H2/CH4
    "R4_C2H6_C2H2": 0.40,  # C2H6/C2H2
}

# Fault criteria: True means the ratio must be >= critical value
# Each entry: (R1_high, R2_high, R3_high, R4_high)
# True  = ratio must be ≥ critical value
# False = ratio must be <  critical value
_FAULT_CRITERIA: dict[str, tuple[bool, bool, bool, bool]] = {
    "T":  (True,  False, False, True),   # Thermal
    "D":  (False, True,  True,  False),  # Arcing / Electrical discharge
    "PD": (False, False, False, True),   # Corona / Partial discharge
    # Note: PD also requires R1 < 0.1 (a stricter sub-threshold handled below)
}

_FAULT_FULL_NAMES = {
    "T":            "Thermal Fault",
    "D":            "Electrical Discharge (Arcing)",
    "PD":           "Partial Discharge (Corona)",
    "Undetermined": "Undetermined",
}

_FAULT_DESCRIPTIONS = {
    "T":  "Thermal decomposition of oil and/or paper insulation (overheating). "
          "Maps to T1/T2/T3 depending on severity — use Duval Triangle to sub-classify.",
    "D":  "High-energy electrical discharge (arcing) — D2 most likely. "
          "Urgent: take transformer offline for inspection.",
    "PD": "Low-energy partial discharge or corona — early-stage electrical stress. "
          "Monitor closely; trending is more important than a single reading.",
    "Undetermined": "Ratio pattern does not clearly match any fault type. "
                    "Cross-reference with other DGA methods.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def doernenburg(
    H2: float,
    CH4: float,
    C2H6: float,
    C2H4: float,
    C2H2: float,
    eps: float = 1e-6,
) -> dict:
    """
    Diagnose a DGA sample using the Doernenburg Ratio Method.

    Parameters
    ----------
    H2, CH4, C2H6, C2H4, C2H2 : float
        Dissolved gas concentrations in ppm (µL/L).
    eps : float
        Small constant added to denominators to prevent ZeroDivisionError.

    Returns
    -------
    dict
        Keys:
        - method         (str)  : "Doernenburg"
        - diagnosis      (str)  : fault class code ("T", "D", "PD",
                                  "Undetermined") or "N/A"
        - diagnosis_full (str)  : expanded fault name
        - description    (str)  : plain-language explanation
        - applicable     (bool) : False if Stage-1 concentrations not exceeded
        - stage1         (dict) : which gases exceeded L1 threshold
        - ratios         (dict) : computed R1–R4 values
        - scores         (dict) : how many criteria each fault type satisfied
        - note           (str)  : interpretive note for the engineer

    Example
    -------
    >>> from doernenburg import doernenburg
    >>> result = doernenburg(H2=430, CH4=200, C2H6=80, C2H4=380, C2H2=210)
    >>> print(result["diagnosis_full"])
    Electrical Discharge (Arcing)
    """
    # ── Stage 1: minimum concentration check ──────────────────────────────
    stage1 = {
        "H2_ok":   H2   > _L1["H2"],
        "C2H2_ok": C2H2 > _L1["C2H2"],
        "CH4_ok":  CH4  > _L1["CH4"],
        "C2H4_ok": C2H4 > _L1["C2H4"],
    }
    any_exceeded = any(stage1.values())

    if not any_exceeded:
        return {
            "method":         "Doernenburg",
            "diagnosis":      "N/A",
            "diagnosis_full": "Not Applicable",
            "description": (
                "No sentinel gas exceeded its Stage-1 minimum threshold "
                f"(H2>{_L1['H2']}, C2H2>{_L1['C2H2']}, "
                f"CH4>{_L1['CH4']}, C2H4>{_L1['C2H4']} ppm). "
                "Gas concentrations are consistent with normal oil aging. "
                "Continue routine monitoring per IEC 60599 schedule."
            ),
            "applicable": False,
            "stage1":     stage1,
            "ratios":     {},
            "scores":     {},
            "note":       "Doernenburg method not applicable — gas levels too low.",
        }

    # ── Stage 2: compute ratios ────────────────────────────────────────────
    R1 = CH4  / (H2   + eps)
    R2 = C2H2 / (C2H4 + eps)
    R3 = C2H2 / (CH4  + eps)
    R4 = C2H6 / (C2H2 + eps)

    ratios = {
        "R1 (CH4/H2)":    round(R1, 4),
        "R2 (C2H2/C2H4)": round(R2, 4),
        "R3 (C2H2/CH4)":  round(R3, 4),
        "R4 (C2H6/C2H2)": round(R4, 4),
    }

    # Boolean flags: is each ratio ≥ its critical value?
    r_high = (
        R1 >= _CRITICAL["R1_CH4_H2"],
        R2 >= _CRITICAL["R2_C2H2_C2H4"],
        R3 >= _CRITICAL["R3_C2H2_CH4"],
        R4 >= _CRITICAL["R4_C2H6_C2H2"],
    )

    # ── Score each fault type ──────────────────────────────────────────────
    scores: dict[str, int] = {}
    for fault, criteria in _FAULT_CRITERIA.items():
        score = sum(
            1 for req, actual_high in zip(criteria, r_high)
            if req == actual_high
        )
        # Extra check for PD: R1 must also be < 0.1 (strict corona criterion)
        if fault == "PD" and R1 >= 0.1:
            score -= 1  # penalise if R1 is not in the low-corona range
        scores[fault] = score

    # ── Pick diagnosis ─────────────────────────────────────────────────────
    best_fault  = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_fault]

    if best_score >= 3:
        diagnosis = best_fault
    else:
        diagnosis = "Undetermined"

    note = (
        f"Stage-1 passed ({sum(stage1.values())} gas(es) exceeded threshold). "
        f"Ratio scores — T:{scores.get('T',0)}, D:{scores.get('D',0)}, "
        f"PD:{scores.get('PD',0)}.  "
        f"Minimum 3 needed for a definitive call; "
        f"best score = {best_score}."
    )

    return {
        "method":         "Doernenburg",
        "diagnosis":      diagnosis,
        "diagnosis_full": _FAULT_FULL_NAMES[diagnosis],
        "description":    _FAULT_DESCRIPTIONS[diagnosis],
        "applicable":     True,
        "stage1":         stage1,
        "ratios":         ratios,
        "scores":         scores,
        "note":           note,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run with: python doernenburg.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # (H2,  CH4,  C2H6, C2H4, C2H2, label)
        (10,   5,    2,    3,    1,   "Normal — below L1, N/A expected"),
        (430,  200,  80,   380,  210, "Arcing — D expected"),
        (150,  250,  90,   130,  10,  "Thermal — T expected"),
        (950,  80,   30,   25,   8,   "PD/Corona — PD expected"),
    ]

    print("=" * 70)
    print("Doernenburg Method — Self-Test")
    print("Reference: IEEE C57.104-2019, Annex C | IEC 60599:2022, Annex B")
    print("=" * 70)

    for H2, CH4, C2H6, C2H4, C2H2, label in test_cases:
        r = doernenburg(H2, CH4, C2H6, C2H4, C2H2)
        print(f"\nTest : {label}")
        print(f"  Gases  : H2={H2}, CH4={CH4}, C2H6={C2H6}, C2H4={C2H4}, C2H2={C2H2}")
        print(f"  Result : {r['diagnosis']} — {r['diagnosis_full']}")
        print(f"  Note   : {r['note']}")

    print("\n" + "=" * 70)
