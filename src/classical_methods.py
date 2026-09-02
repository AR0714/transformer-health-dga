# -*- coding: utf-8 -*-
"""
Classical DGA Diagnostic Methods
=================================
Four industry-standard rule-based methods for transformer fault diagnosis.

References
----------
Key Gas      : IEEE C57.104-2019 Table 6
Rogers       : IEC 60599:2022 Annex C; Rogers (1978)
Doernenburg  : IEEE C57.104-1991 Table 2; Doernenburg & Strittmatter (1967)
Duval Tri.   : IEC 60599:2022 Annex D; Duval (1974, 2002)
"""

from collections import Counter

FAULT_CLASSES = ["Normal", "PD", "D1", "D2", "T1", "T2", "T3"]


def key_gas_method(H2, CH4, C2H6, C2H4, C2H2):
    """IEEE C57.104-2019 Key Gas Method.

    Each fault type produces a characteristic dominant gas.
    Priority: C2H2 (arcing) > H2 (PD) > C2H4 (high thermal) > CH4 (low thermal).
    """
    eps = 1e-6
    total = H2 + CH4 + C2H6 + C2H4 + C2H2
    if total < 10:
        return "Normal"

    # C2H2 at any significant level signals discharge / arcing
    if C2H2 > 35:
        if C2H2 / (C2H4 + eps) > 0.75:
            return "D1"   # C2H2 dominant -> low-energy arc
        return "D2"       # C2H2 + C2H4 mix -> high-energy arc

    # H2 dominant -> partial discharge
    if H2 > max(CH4, C2H4, C2H2) * 2 and H2 > 100:
        return "PD"

    # C2H4 dominant -> thermal fault
    if C2H4 > CH4 * 1.5 and C2H4 > H2:
        if C2H4 > 200 or C2H4 / (C2H6 + eps) > 3:
            return "T3"   # Very high C2H4 -> >700 deg C
        return "T2"       # Moderate C2H4 -> 300-700 deg C

    # CH4 / C2H6 dominant -> low thermal
    if CH4 > C2H4 and CH4 + C2H6 > H2:
        return "T1"       # <300 deg C

    return "Normal"


def rogers_ratios_method(H2, CH4, C2H6, C2H4, C2H2):
    """Rogers 3-Ratio Method.

    Reference: IEC 60599:2022 Annex C; Rogers (1978)
    Ratios: R1 = C2H2/C2H4, R2 = CH4/H2, R3 = C2H4/C2H6
    Each ratio encodes to a code (0/1/2); codes map to a fault class.
    """
    eps = 1e-6
    R1 = C2H2 / (C2H4 + eps)
    R2 = CH4  / (H2   + eps)
    R3 = C2H4 / (C2H6 + eps)

    c1 = 0 if R1 < 0.1 else (1 if R1 < 3.0 else 2)
    c2 = 1 if R2 < 0.1 else (0 if R2 < 1.0 else 2)
    c3 = 0 if R3 < 1.0 else (1 if R3 < 3.0 else 2)

    # Lookup: key = (c2, c1, c3)
    lookup = {
        (0, 0, 0): "Normal",
        (1, 0, 0): "PD",
        (1, 1, 0): "D1",
        (0, 1, 0): "D1",
        (0, 2, 0): "D2",
        (0, 2, 1): "D2",
        (0, 2, 2): "D2",
        (2, 0, 0): "T1",
        (0, 0, 1): "T1",
        (2, 0, 1): "T2",
        (2, 0, 2): "T3",
        (0, 0, 2): "T3",
        (2, 1, 0): "D1",
        (2, 2, 0): "D2",
    }
    return lookup.get((c2, c1, c3), "Unknown")


def doernenburg_method(H2, CH4, C2H6, C2H4, C2H2):
    """Doernenburg Ratio Method.

    Reference: Doernenburg & Strittmatter (1967); IEEE C57.104-1991 Table 2
    Ratios: R1=CH4/H2, R2=C2H2/C2H4, R3=C2H2/CH4, R4=C2H6/C2H2
    Requires at least one gas to exceed Level-1 concentration limits.
    """
    eps = 1e-6
    LIMITS = {"H2": 100, "CH4": 25, "C2H4": 10, "C2H2": 1, "C2H6": 15}
    gases  = {"H2": H2, "CH4": CH4, "C2H4": C2H4, "C2H2": C2H2, "C2H6": C2H6}
    if not any(v > LIMITS[g] for g, v in gases.items()):
        return "Normal"   # All gases below thresholds

    R1 = CH4  / (H2   + eps)
    R2 = C2H2 / (C2H4 + eps)
    R3 = C2H2 / (CH4  + eps)
    R4 = C2H6 / (C2H2 + eps)

    if R1 < 0.1 and R3 < 0.3 and R4 > 10:
        return "PD"
    if R2 > 0.75 and R3 > 0.3:
        return "D1"
    if 0.1 <= R2 <= 0.75 and R3 > 0.3:
        return "D2"
    if R1 > 1.0 and R2 < 0.1 and R3 < 0.3:
        if R4 > 4.0:
            return "T1"
        if R4 >= 0.4:
            return "T2"
        return "T3"
    return "Unknown"


def duval_triangle_method(H2, CH4, C2H6, C2H4, C2H2):
    """Duval Triangle Method.

    Reference: IEC 60599:2022 Annex D; Duval (1974, 2002)
    Projects CH4, C2H4, C2H2 onto a ternary diagram and identifies the fault zone.
    Zone boundaries follow Duval (2002), IEEE Electr. Insul. Magazine.
    """
    eps = 1e-6
    total = CH4 + C2H4 + C2H2
    if total < eps:
        return "Normal"

    p = 100 * CH4  / total   # %CH4
    q = 100 * C2H4 / total   # %C2H4
    r = 100 * C2H2 / total   # %C2H2

    # Discharge zones (significant C2H2)
    if r >= 29:
        return "D1"
    if 13 <= r < 29:
        return "D2" if q >= 23 else "D1"
    if 4 <= r < 13:
        return "D2" if q >= 25 else "T3"   # DT mixed zone

    # Thermal zones (r < 4, negligible C2H2)
    if q >= 50:
        return "T3"
    if q >= 20:
        return "T2"
    if q >= 4:
        return "T1"
    if r < 2 and q < 4:
        return "PD"
    return "T1"


def run_all_classical(H2, CH4, C2H6, C2H4, C2H2):
    """Run all four classical methods and return a results dict."""
    return {
        "Key Gas":     key_gas_method(H2, CH4, C2H6, C2H4, C2H2),
        "Rogers":      rogers_ratios_method(H2, CH4, C2H6, C2H4, C2H2),
        "Doernenburg": doernenburg_method(H2, CH4, C2H6, C2H4, C2H2),
        "Duval":       duval_triangle_method(H2, CH4, C2H6, C2H4, C2H2),
    }


def classical_consensus(H2, CH4, C2H6, C2H4, C2H2):
    """Majority-vote consensus across all classical methods (Unknown excluded)."""
    results = run_all_classical(H2, CH4, C2H6, C2H4, C2H2)
    valid = [v for v in results.values() if v != "Unknown"]
    if not valid:
        return "Unknown"
    return Counter(valid).most_common(1)[0][0]
