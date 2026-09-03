# -*- coding: utf-8 -*-
# predict.py -- Transformer fault diagnosis from DGA gas readings
# Usage: python predict.py <H2> <CH4> <C2H6> <C2H4> <C2H2>
import sys, os, joblib
import numpy as np

ABSTAIN_THRESHOLD = 0.50

# Physical plausibility limits (ppm) -- values above these are sensor errors
GAS_LIMITS = {
    "H2":   50000,
    "CH4":  10000,
    "C2H6":  5000,
    "C2H4":  5000,
    "C2H2":  5000,
}


def validate_gases(H2, CH4, C2H6, C2H4, C2H2):
    """Validate gas inputs. Returns list of error strings (empty = all OK)."""
    gases = {"H2": H2, "CH4": CH4, "C2H6": C2H6, "C2H4": C2H4, "C2H2": C2H2}
    errors = []
    warnings = []

    # Rule 1: No negative values
    for name, val in gases.items():
        if val < 0:
            errors.append(f"  ERROR: {name} = {val} ppm is negative (gas concentrations must be >= 0)")

    # Rule 2: Total gas must be non-zero
    if sum(gases.values()) == 0:
        errors.append("  ERROR: All gas values are zero -- no dissolved gas to analyse")

    # Rule 3: Flag physically implausible values
    for name, val in gases.items():
        if val > GAS_LIMITS[name]:
            warnings.append(f"  WARNING: {name} = {val:.0f} ppm exceeds plausibility limit "
                            f"({GAS_LIMITS[name]} ppm) -- check sensor / units")

    return errors, warnings


def main():
    if len(sys.argv) != 6:
        print("Usage: python predict.py <H2> <CH4> <C2H6> <C2H4> <C2H2>")
        print("       All values in ppm (micro-litres per litre of oil)")
        sys.exit(1)

    try:
        H2, CH4, C2H6, C2H4, C2H2 = map(float, sys.argv[1:6])
    except ValueError:
        print("ERROR: All five arguments must be numbers.")
        sys.exit(1)

    # ── Input validation ──────────────────────────────────────────────────────
    errors, warnings = validate_gases(H2, CH4, C2H6, C2H4, C2H2)
    if errors:
        print("\n=== Input Validation Failed ===")
        for e in errors:
            print(e)
        print("Exiting -- fix the inputs and retry.")
        sys.exit(1)
    if warnings:
        print("\n=== Input Warnings ===")
        for w in warnings:
            print(w)
        print("Proceeding with diagnosis, but verify sensor readings.")

    # ── Load model and predict ────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_dga_calibrated.joblib')
    cal_model = joblib.load(model_path)
    result = cal_model.predict_sample(H2=H2, CH4=CH4, C2H6=C2H6, C2H4=C2H4, C2H2=C2H2)

    print("\n=== DGA Fault Diagnosis ===")
    print(f"Gases (ppm)  : H2={H2}, CH4={CH4}, C2H6={C2H6}, C2H4={C2H4}, C2H2={C2H2}")
    print(f"Diagnosis    : {result['diagnosis']}")
    print(f"Confidence   : {result['confidence']:.1%}")
    print(f"Reliability  : {result['reliability']}")
    if result['confidence'] < ABSTAIN_THRESHOLD:
        print("WARNING      : Low confidence -- route to engineer review")

    print("\nClass probabilities:")
    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 40)
        print(f"  {cls:<8} {prob:.1%}  {bar}")
    print("===========================\n")


if __name__ == '__main__':
    main()
