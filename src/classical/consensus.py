"""
Multi-Method DGA Consensus Engine
===================================
Reference : Sutikno, T. et al. (2024).
            Multi-method DGA interpretation for power transformer fault diagnosis.
            Heliyon, 10(4), e25975.

            Camlin Group — TOTUS G9 DGA Monitor product description:
            "Simplifies DGA interpretation by bringing all leading diagnostic
            methods together in one clear, colour-coded view."
            (Cited as industrial precedent for consensus-based DGA systems.)

Purpose
-------
In real transformer diagnostics, a single classical method is never used
in isolation.  Engineers cross-reference multiple methods because:

  1. Each method has "blind spots" — ratio codes that don't map to a fault.
  2. Methods disagree at fault-type boundaries (e.g. D1 vs D2).
  3. Consensus across methods gives higher confidence than any single method.

This module runs all SIX diagnostic methods simultaneously on one gas sample:
    Classical:  Key Gas · IEC 60599 3-Ratio · Duval Triangle 1
    New:        Rogers Ratio · Doernenburg · Duval Pentagon 1

It then computes an agreement score and compares the consensus against
an optional ML model result.

Output schema
-------------
{
    "sample"    : {H2, CH4, C2H6, C2H4, C2H2},
    "results"   : {
        "key_gas"     : { method, diagnosis, applicable, ... },
        "iec_ratios"  : { method, diagnosis, applicable, ... },
        "duval"       : { method, diagnosis, applicable, ... },
        "rogers"      : { method, diagnosis, applicable, ... },
        "doernenburg" : { method, diagnosis, applicable, ... },
        "pentagon"    : { method, diagnosis, applicable, ... },
    },
    "consensus" : {
        "all_diagnoses"      : [...],   # from applicable methods only
        "plurality_vote"     : "D2",    # most common diagnosis
        "agreement_count"    : 5,       # how many methods agree on plurality
        "total_applicable"   : 5,       # methods that returned a result
        "agreement_pct"      : 100.0,   # agreement_count / total_applicable * 100
        "is_split"           : False,   # True if top two diagnoses are tied
        "second_diagnosis"   : None,    # second-most-common diagnosis (if split)
    },
    "ml_comparison" : {
        "ml_diagnosis"       : "D2",    # or None if not provided
        "ml_confidence"      : 0.87,    # or None if not provided
        "matches_consensus"  : True,    # None if no ML result given
        "agreement_note"     : "...",   # plain-language comment
    },
    "summary" : "...",   # one-paragraph plain-English summary for the engineer
}
"""

import sys
import os
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Import the six diagnostic method modules
# ---------------------------------------------------------------------------
# Allow running this file from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Adapter wrappers for Part-1 methods
# Part-1 functions use varied return formats; these wrappers normalise them
# to the standard schema: {method, diagnosis, applicable} used by the engine.
# ---------------------------------------------------------------------------

def _key_gas(H2: float, CH4: float, C2H6: float, C2H4: float, C2H2: float) -> dict:
    """
    Adapter for Part-1 key_gas.key_gas().
    Part-1 returns a plain string (e.g. "PD").
    """
    try:
        from key_gas import key_gas as _func
        result = _func(H2, CH4, C2H6, C2H4, C2H2)
        if isinstance(result, str):
            return {"method": "Key Gas", "diagnosis": result, "applicable": True}
        if isinstance(result, dict):
            diag = result.get("diagnosis", result.get("fault", "Undetermined"))
            return {"method": "Key Gas", "diagnosis": diag, "applicable": True}
    except Exception as exc:
        return {"method": "Key Gas", "diagnosis": "Error", "applicable": False,
                "note": str(exc)}
    return {"method": "Key Gas", "diagnosis": "Undetermined", "applicable": True}


def _iec_ratios(H2: float, CH4: float, C2H6: float, C2H4: float, C2H2: float) -> dict:
    """
    Adapter for Part-1 iec_ratios.iec_ratios().
    Part-1 returns a dict with key 'fault' (not 'diagnosis').
    "No decision (outside IEC table)" is mapped to "Undetermined".
    """
    try:
        from iec_ratios import iec_ratios as _func
        result = _func(H2, CH4, C2H6, C2H4, C2H2)
        if isinstance(result, dict):
            raw = result.get("fault", result.get("diagnosis", "Undetermined"))
            if raw is None or "No decision" in str(raw) or "outside" in str(raw).lower():
                raw = "Undetermined"
            return {"method": "IEC 60599 3-Ratio", "diagnosis": raw, "applicable": True}
        if isinstance(result, str):
            return {"method": "IEC 60599 3-Ratio", "diagnosis": result, "applicable": True}
    except Exception as exc:
        return {"method": "IEC 60599 3-Ratio", "diagnosis": "Error", "applicable": False,
                "note": str(exc)}
    return {"method": "IEC 60599 3-Ratio", "diagnosis": "Undetermined", "applicable": True}


def _duval(H2: float, CH4: float, C2H6: float, C2H4: float, C2H2: float,
           eps: float = 1e-6) -> dict:
    """
    Adapter for Part-1 duval.duval_zone().
    Part-1 duval_zone() takes 3 percentage arguments (pCH4, pC2H4, pC2H2),
    not raw gas concentrations.  This wrapper computes the percentages first.
    """
    try:
        from duval import duval_zone as _zone
        total = CH4 + C2H4 + C2H2 + eps
        pCH4  = CH4  / total * 100.0
        pC2H4 = C2H4 / total * 100.0
        pC2H2 = C2H2 / total * 100.0
        result = _zone(pCH4, pC2H4, pC2H2)
        if isinstance(result, str):
            return {"method": "Duval Triangle 1", "diagnosis": result, "applicable": True}
        if isinstance(result, dict):
            diag = result.get("diagnosis", result.get("fault", "Undetermined"))
            return {"method": "Duval Triangle 1", "diagnosis": diag, "applicable": True}
    except Exception as exc:
        return {"method": "Duval Triangle 1", "diagnosis": "Error", "applicable": False,
                "note": str(exc)}
    return {"method": "Duval Triangle 1", "diagnosis": "Undetermined", "applicable": True}

# New Part-2 methods
try:
    from rogers      import rogers_ratio    as _rogers
    from doernenburg import doernenburg     as _doernenburg
    from pentagon    import duval_pentagon  as _pentagon
except ImportError as e:
    raise ImportError(
        "Could not import Part-2 classical methods. "
        "Make sure rogers.py, doernenburg.py, and pentagon.py are present.\n"
        f"Original error: {e}"
    )

__all__ = ["run_consensus"]


# ---------------------------------------------------------------------------
# Fault severity ranking (for tie-breaking and reporting)
# ---------------------------------------------------------------------------
_SEVERITY = {
    "Normal": 0,
    "PD":     1,
    "T1":     2,
    "T2":     3,
    "D1":     3,
    "T3":     4,
    "DT":     4,   # Duval mixed Discharge-Thermal zone (Part-1)
    "D2":     5,
    "T":      3,   # Doernenburg broad thermal class
    "D":      5,   # Doernenburg broad discharge class (equivalent to D2)
}

# Labels that mean "the method gave no definitive answer"
_INCONCLUSIVE = {"Undetermined", "N/A", "Not Applicable", None}

# Colour coding for console output (ANSI escape codes)
_COLOUR = {
    "Normal": "\033[92m",   # green
    "PD":     "\033[96m",   # cyan
    "D1":     "\033[93m",   # yellow
    "D2":     "\033[91m",   # red
    "T1":     "\033[93m",   # yellow
    "T2":     "\033[33m",   # orange-ish
    "T3":     "\033[91m",   # red
    "DT":     "\033[33m",   # orange — mixed discharge-thermal (Part-1 Duval)
    "T":      "\033[33m",   # orange — Doernenburg broad thermal
    "D":      "\033[91m",   # red   — Doernenburg broad discharge
}
_RESET = "\033[0m"


def _colour(diagnosis: str) -> str:
    col = _COLOUR.get(diagnosis, "")
    return f"{col}{diagnosis}{_RESET}" if col else diagnosis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _safe_call(func, H2, CH4, C2H6, C2H4, C2H2):
    """Call a classical method and catch any unexpected exception."""
    try:
        return func(H2=H2, CH4=CH4, C2H6=C2H6, C2H4=C2H4, C2H2=C2H2)
    except Exception as exc:
        return {
            "method":      func.__name__,
            "diagnosis":   "Error",
            "applicable":  False,
            "note":        f"Method raised exception: {exc}",
        }


def _compute_consensus(results: dict) -> dict:
    """Aggregate results from all methods into a consensus block."""
    all_diagnoses = []

    for key, r in results.items():
        diagnosis = r.get("diagnosis", None)
        applicable = r.get("applicable", True)
        if applicable and diagnosis not in _INCONCLUSIVE:
            all_diagnoses.append(diagnosis)

    if not all_diagnoses:
        return {
            "all_diagnoses":    [],
            "plurality_vote":   "Undetermined",
            "agreement_count":  0,
            "total_applicable": 0,
            "agreement_pct":    0.0,
            "is_split":         False,
            "second_diagnosis": None,
        }

    counts = Counter(all_diagnoses)
    most_common = counts.most_common()
    top_diagnosis  = most_common[0][0]
    top_count      = most_common[0][1]
    total          = len(all_diagnoses)
    agreement_pct  = (top_count / total) * 100.0

    # Check for a tie at the top
    is_split = len(most_common) > 1 and most_common[1][1] == top_count
    second   = most_common[1][0] if is_split else (
        most_common[1][0] if len(most_common) > 1 else None
    )

    return {
        "all_diagnoses":    all_diagnoses,
        "plurality_vote":   top_diagnosis,
        "agreement_count":  top_count,
        "total_applicable": total,
        "agreement_pct":    round(agreement_pct, 1),
        "is_split":         is_split,
        "second_diagnosis": second,
    }


def _ml_comparison_block(
    consensus_diagnosis: str,
    ml_diagnosis: Optional[str],
    ml_confidence: Optional[float],
) -> dict:
    """Build the ML comparison sub-block."""
    if ml_diagnosis is None:
        return {
            "ml_diagnosis":    None,
            "ml_confidence":   None,
            "matches_consensus": None,
            "agreement_note":  "No ML model result provided for this sample.",
        }

    matches = (ml_diagnosis == consensus_diagnosis)
    conf_str = f" (confidence {ml_confidence:.1%})" if ml_confidence is not None else ""

    if matches:
        note = (
            f"ML model agrees with classical consensus: both diagnose {ml_diagnosis}"
            f"{conf_str}. High confidence result."
        )
    else:
        sev_ml  = _SEVERITY.get(ml_diagnosis, -1)
        sev_con = _SEVERITY.get(consensus_diagnosis, -1)
        if sev_ml > sev_con:
            note = (
                f"ML model ({ml_diagnosis}{conf_str}) gives a MORE severe diagnosis "
                f"than classical consensus ({consensus_diagnosis}). "
                "Consider the ML result — it may have detected subtle gas patterns "
                "that threshold-based methods miss."
            )
        else:
            note = (
                f"ML model ({ml_diagnosis}{conf_str}) gives a LESS severe diagnosis "
                f"than classical consensus ({consensus_diagnosis}). "
                "Review gas concentrations carefully; classical methods may be "
                "capturing an interaction the ML model underweighted."
            )

    return {
        "ml_diagnosis":      ml_diagnosis,
        "ml_confidence":     ml_confidence,
        "matches_consensus": matches,
        "agreement_note":    note,
    }


def _build_summary(
    sample: dict,
    consensus: dict,
    ml_block: dict,
) -> str:
    """Generate a one-paragraph plain-English summary for an engineer."""
    pv  = consensus["plurality_vote"]
    ag  = consensus["agreement_count"]
    tot = consensus["total_applicable"]
    pct = consensus["agreement_pct"]

    lines = [
        f"DGA CONSENSUS SUMMARY",
        f"Sample gases (ppm): H2={sample['H2']}, CH4={sample['CH4']}, "
        f"C2H6={sample['C2H6']}, C2H4={sample['C2H4']}, C2H2={sample['C2H2']}.",
        "",
        f"Result: {ag} of {tot} applicable methods ({pct:.0f}% agreement) "
        f"diagnose this sample as '{pv}'.",
    ]

    if consensus["is_split"]:
        lines.append(
            f"NOTE: There is a split — '{consensus['second_diagnosis']}' "
            "received equal votes. Treat this as an ambiguous boundary case."
        )

    ml = ml_block
    if ml["ml_diagnosis"] is not None:
        conf = f" ({ml['ml_confidence']:.1%})" if ml["ml_confidence"] else ""
        lines.append(
            f"ML model diagnosis: {ml['ml_diagnosis']}{conf}. "
            f"{'Agrees with' if ml['matches_consensus'] else 'DIFFERS FROM'} "
            "classical consensus."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_consensus(
    H2:   float,
    CH4:  float,
    C2H6: float,
    C2H4: float,
    C2H2: float,
    ml_diagnosis:  Optional[str]   = None,
    ml_confidence: Optional[float] = None,
) -> dict:
    """
    Run all six DGA diagnostic methods and compute a consensus result.

    Parameters
    ----------
    H2, CH4, C2H6, C2H4, C2H2 : float
        Dissolved gas concentrations in ppm (µL/L).
    ml_diagnosis : str, optional
        The fault class predicted by the ML model (e.g. "D2").
        If None, the ML comparison block will show "not provided".
    ml_confidence : float, optional
        Calibrated probability for ml_diagnosis (0.0–1.0).

    Returns
    -------
    dict
        Full consensus result — see module docstring for schema.

    Example
    -------
    >>> from consensus import run_consensus
    >>> result = run_consensus(
    ...     H2=430, CH4=95, C2H6=30, C2H4=380, C2H2=210,
    ...     ml_diagnosis="D2", ml_confidence=0.87
    ... )
    >>> print(result["consensus"]["plurality_vote"])
    D2
    >>> print(result["consensus"]["agreement_pct"])
    100.0
    """
    sample = {
        "H2": H2, "CH4": CH4, "C2H6": C2H6,
        "C2H4": C2H4, "C2H2": C2H2,
    }

    # ── Run all six methods ────────────────────────────────────────────────
    results = {
        "key_gas":     _safe_call(_key_gas,     H2, CH4, C2H6, C2H4, C2H2),
        "iec_ratios":  _safe_call(_iec_ratios,  H2, CH4, C2H6, C2H4, C2H2),
        "duval":       _safe_call(_duval,        H2, CH4, C2H6, C2H4, C2H2),
        "rogers":      _safe_call(_rogers,       H2, CH4, C2H6, C2H4, C2H2),
        "doernenburg": _safe_call(_doernenburg,  H2, CH4, C2H6, C2H4, C2H2),
        "pentagon":    _safe_call(_pentagon,     H2, CH4, C2H6, C2H4, C2H2),
    }

    # ── Aggregate ──────────────────────────────────────────────────────────
    consensus  = _compute_consensus(results)
    ml_block   = _ml_comparison_block(
        consensus["plurality_vote"], ml_diagnosis, ml_confidence
    )
    summary    = _build_summary(sample, consensus, ml_block)

    return {
        "sample":        sample,
        "results":       results,
        "consensus":     consensus,
        "ml_comparison": ml_block,
        "summary":       summary,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_consensus(result: dict) -> None:
    """Print a formatted consensus report to stdout."""
    print("\n" + "═" * 68)
    print("  DGA MULTI-METHOD CONSENSUS ENGINE")
    print("  All 6 Methods · IEC 60599 | IEEE C57.104 | Duval 2014")
    print("═" * 68)

    s = result["sample"]
    print(f"\n  Sample (ppm):  H2={s['H2']}  CH4={s['CH4']}  "
          f"C2H6={s['C2H6']}  C2H4={s['C2H4']}  C2H2={s['C2H2']}")

    print(f"\n  {'Method':<25} {'Diagnosis':<16} {'Applicable'}")
    print("  " + "-" * 50)

    labels = {
        "key_gas":     "Key Gas Method",
        "iec_ratios":  "IEC 60599 3-Ratio",
        "duval":       "Duval Triangle 1",
        "rogers":      "Rogers Ratio",
        "doernenburg": "Doernenburg",
        "pentagon":    "Duval Pentagon 1",
    }

    for key, label in labels.items():
        r = result["results"][key]
        diag = r.get("diagnosis", "—")
        appl = "✓" if r.get("applicable", True) else "✗ N/A"
        print(f"  {label:<25} {_colour(diag):<16} {appl}")

    c = result["consensus"]
    print(f"\n  {'─'*50}")
    print(f"  Consensus (plurality vote) : {_colour(c['plurality_vote'])}")
    print(f"  Agreement                  : {c['agreement_count']}/{c['total_applicable']} "
          f"methods  ({c['agreement_pct']:.0f}%)")

    if c["is_split"]:
        print(f"  ⚠  Split result — tied with : {c['second_diagnosis']}")

    ml = result["ml_comparison"]
    if ml["ml_diagnosis"] is not None:
        conf_str = f" ({ml['ml_confidence']:.1%})" if ml["ml_confidence"] else ""
        match_str = "✓ agrees" if ml["matches_consensus"] else "✗ DIFFERS"
        print(f"\n  ML model : {_colour(ml['ml_diagnosis'])}{conf_str}  →  {match_str}")
        print(f"  {ml['agreement_note']}")

    print("\n" + "═" * 68 + "\n")


# ---------------------------------------------------------------------------
# Quick self-test (run with: python consensus.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Representative arcing sample (D2 expected across all methods)
    result = run_consensus(
        H2=430, CH4=95, C2H6=30, C2H4=380, C2H2=210,
        ml_diagnosis="D2", ml_confidence=0.87,
    )
    print_consensus(result)
    print(result["summary"])
    print()

    # Thermal sample (T3 expected)
    result2 = run_consensus(
        H2=40, CH4=210, C2H6=60, C2H4=290, C2H2=5,
        ml_diagnosis="T3", ml_confidence=0.79,
    )
    print_consensus(result2)
