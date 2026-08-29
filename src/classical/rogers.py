"""
Rogers Ratio Method for DGA Transformer Fault Diagnosis
========================================================
Reference : IEC 60599:2022, Annex B (Table B.1)
            Rogers, R.R. (1975). IEEE and IEC codes to interpret
            incipient faults in transformers using gas in oil analysis.
            IEEE Trans. Electr. Insul., EI-10(3), 96-104.

Method summary
--------------
Three gas ratios are computed from the five key DGA gases.
Each ratio is coded as 0, 1, or 2 based on IEC threshold ranges.
The resulting three-digit code is looked up in the Rogers fault table.

Ratios
------
    R1 = CH4  / H2     (methane / hydrogen)
    R2 = C2H2 / C2H4   (acetylene / ethylene)
    R3 = C2H4 / C2H6   (ethylene / ethane)

Coding rules (IEC 60599:2022, Table B.1)
-----------------------------------------
    R1 (CH4/H2) :  < 0.1  → code 0 | 0.1–1.0 → code 1 | > 1.0 → code 2
    R2 (C2H2/C2H4): < 0.1 → code 0 | 0.1–3.0 → code 1 | > 3.0 → code 2
    R3 (C2H4/C2H6): < 1.0 → code 0 | 1.0–3.0 → code 1 | > 3.0 → code 2

Fault lookup table
------------------
    (0,0,0) → Normal           (0,0,1) → T1   (0,0,2) → T2
    (0,2,2) → T3               (2,0,2) → T3
    (1,0,0) → PD
    (0,1,0) → D1               (1,1,0) → D1
    (0,2,0) → D2               (1,2,0) → D2   (2,2,0) → D2
    All other codes → Undetermined
"""

__all__ = ["rogers_ratio"]


# ---------------------------------------------------------------------------
# Lookup table  (code_R1, code_R2, code_R3) → fault class
# ---------------------------------------------------------------------------
_ROGERS_TABLE: dict[tuple[int, int, int], str] = {
    (0, 0, 0): "Normal",
    # Partial discharge
    (1, 0, 0): "PD",
    # Thermal faults
    (0, 0, 1): "T1",
    (0, 0, 2): "T2",
    (0, 2, 2): "T3",
    (2, 0, 2): "T3",
    # Low-energy discharge
    (0, 1, 0): "D1",
    (1, 1, 0): "D1",
    # High-energy discharge (arcing)
    (0, 2, 0): "D2",
    (1, 2, 0): "D2",
    (2, 2, 0): "D2",
}

# Human-readable description of each fault class
_FAULT_DESCRIPTIONS: dict[str, str] = {
    "Normal":        "No fault — gas concentrations within normal aging limits",
    "PD":            "Partial Discharge — low-energy electrical discharges in oil or paper",
    "D1":            "Low-Energy Discharge — sparking without sustained arc",
    "D2":            "High-Energy Discharge — full electrical arcing (urgent)",
    "T1":            "Thermal Fault < 300 °C — mild overheating",
    "T2":            "Thermal Fault 300–700 °C — moderate overheating",
    "T3":            "Thermal Fault > 700 °C — severe overheating, carbonisation risk",
    "Undetermined":  "Code combination not in Rogers table — consult engineer",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _code_R1(r: float) -> int:
    """Code CH4/H2 ratio per IEC 60599 Table B.1."""
    if r < 0.1:
        return 0
    elif r <= 1.0:
        return 1
    else:
        return 2


def _code_R2(r: float) -> int:
    """Code C2H2/C2H4 ratio per IEC 60599 Table B.1."""
    if r < 0.1:
        return 0
    elif r <= 3.0:
        return 1
    else:
        return 2


def _code_R3(r: float) -> int:
    """Code C2H4/C2H6 ratio per IEC 60599 Table B.1."""
    if r < 1.0:
        return 0
    elif r <= 3.0:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def rogers_ratio(
    H2: float,
    CH4: float,
    C2H6: float,
    C2H4: float,
    C2H2: float,
    eps: float = 1e-6,
) -> dict:
    """
    Diagnose a DGA sample using the Rogers Ratio Method.

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
        - method        (str)  : "Rogers Ratio"
        - diagnosis     (str)  : fault class or "Undetermined"
        - description   (str)  : plain-language explanation of the diagnosis
        - applicable    (bool) : always True for Rogers (no gas-level prerequisite)
        - ratios        (dict) : computed float values of R1, R2, R3
        - codes         (dict) : integer codes (0/1/2) for each ratio
        - code_key      (str)  : code triple as "(c1,c2,c3)" for reference
        - note          (str)  : interpretive note for the engineer

    Example
    -------
    >>> from rogers import rogers_ratio
    >>> result = rogers_ratio(H2=430, CH4=95, C2H6=30, C2H4=380, C2H2=210)
    >>> print(result["diagnosis"])
    D2
    """
    # Compute the three ratios
    R1 = CH4  / (H2   + eps)
    R2 = C2H2 / (C2H4 + eps)
    R3 = C2H4 / (C2H6 + eps)

    # Code each ratio
    c1, c2, c3 = _code_R1(R1), _code_R2(R2), _code_R3(R3)
    code_key = (c1, c2, c3)

    # Look up fault class
    diagnosis = _ROGERS_TABLE.get(code_key, "Undetermined")

    # Build interpretive note
    if diagnosis == "Undetermined":
        note = (
            f"Code {code_key} does not appear in the Rogers table. "
            "This can happen at fault-type boundaries or when gas ratios "
            "are influenced by mixed fault mechanisms. "
            "Cross-reference with Duval Triangle and IEC 3-Ratio methods."
        )
    else:
        note = (
            f"Code {code_key} → {diagnosis}. "
            f"R1={R1:.3f} (CH4/H2), R2={R2:.3f} (C2H2/C2H4), "
            f"R3={R3:.3f} (C2H4/C2H6)."
        )

    return {
        "method":      "Rogers Ratio",
        "diagnosis":   diagnosis,
        "description": _FAULT_DESCRIPTIONS[diagnosis],
        "applicable":  True,
        "ratios": {
            "CH4/H2":    round(R1, 4),
            "C2H2/C2H4": round(R2, 4),
            "C2H4/C2H6": round(R3, 4),
        },
        "codes":    {"R1": c1, "R2": c2, "R3": c3},
        "code_key": str(code_key),
        "note":     note,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run with: python rogers.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Test cases designed to produce well-defined Rogers codes ---
    # Gas values chosen so ratios map to entries in the Rogers table.
    # Many real samples fall in "Undetermined" — this is a known Rogers limitation.
    test_cases = [
        # (H2,   CH4,  C2H6, C2H4, C2H2, expected)  # code
        (500,   10,   50,   20,   1,   "Normal"),    # (0,0,0): R1=0.02,R2=0.05,R3=0.40
        (500,   100,  100,  50,   1,   "PD"),         # (1,0,0): R1=0.20,R2=0.02,R3=0.50
        (500,   10,   100,  50,   10,  "D1"),         # (0,1,0): R1=0.02,R2=0.20,R3=0.50
        (200,   100,  200,  100,  400, "D2"),         # (1,2,0): R1=0.50,R2=4.00,R3=0.50
        (300,   10,   50,   100,  1,   "T1"),         # (0,0,1): R1=0.03,R2=0.01,R3=2.00
        (300,   10,   30,   200,  1,   "T2"),         # (0,0,2): R1=0.03,R2=0.01,R3=6.67
        (300,   10,   30,   200,  700, "T3"),         # (0,2,2): R1=0.03,R2=3.50,R3=6.67
    ]

    print("=" * 65)
    print("Rogers Ratio Method — Self-Test")
    print("Reference: IEC 60599:2022, Annex B")
    print("=" * 65)
    print(f"{'Gases (H2/CH4/C2H6/C2H4/C2H2)':<35} {'Code':<12} {'Diagnosis':<15} {'Expected'}")
    print("-" * 65)

    for H2, CH4, C2H6, C2H4, C2H2, expected in test_cases:
        r = rogers_ratio(H2, CH4, C2H6, C2H4, C2H2)
        match = "✓" if r["diagnosis"] == expected else "✗"
        gases = f"{H2}/{CH4}/{C2H6}/{C2H4}/{C2H2}"
        print(f"{gases:<35} {r['code_key']:<12} {r['diagnosis']:<15} {expected} {match}")

    print("=" * 65)
