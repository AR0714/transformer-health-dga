#!/usr/bin/env python
"""predict.py — DGA CLI with validation, SHAP explanation, IEC guidance."""
import argparse, sys, os
import numpy as np

GAS_LIMITS = {"H2":(0,50000),"CH4":(0,50000),"C2H6":(0,20000),
              "C2H4":(0,20000),"C2H2":(0,5000)}

FEAT_NAMES = ["H2","CH4","C2H6","C2H4","C2H2",
              "C2H2/C2H4 ratio","CH4/H2 ratio","C2H4/C2H6 ratio","Total Gas"]

FEAT_PLAIN = {
    "H2":               "Hydrogen (H2) — main PD marker",
    "CH4":              "Methane (CH4) — thermal fault marker",
    "C2H6":             "Ethane (C2H6) — mild thermal marker",
    "C2H4":             "Ethylene (C2H4) — high-temp thermal marker",
    "C2H2":             "Acetylene (C2H2) — arcing/discharge marker",
    "C2H2/C2H4 ratio":  "Acetylene-to-Ethylene ratio (discharge indicator)",
    "CH4/H2 ratio":     "Methane-to-Hydrogen ratio (thermal vs PD separator)",
    "C2H4/C2H6 ratio":  "Ethylene-to-Ethane ratio (temperature severity)",
    "Total Gas":        "Total dissolved gas (overall fault severity)",
}

IEC_GUIDANCE = {
    "Normal": (
        "No active fault. Continue routine monitoring per IEC 60599 Sec 5.1.\n"
        "  Suggested re-test interval: 6-12 months."
    ),
    "PD": (
        "Partial Discharge detected (IEC 60599 Sec 5.3.1).\n"
        "  Action: Check for voids, moisture ingress, or loose metalwork.\n"
        "  Key marker: Hydrogen (H2). Re-test in 1 month."
    ),
    "D1": (
        "Low-energy discharge / sparking (IEC 60599 Sec 5.3.2).\n"
        "  Action: Inspect for carbon tracking, loose connections.\n"
        "  Key markers: C2H2, H2. Consider offline testing."
    ),
    "D2": (
        "High-energy discharge / arcing (IEC 60599 Sec 5.3.3) — URGENT.\n"
        "  Action: Risk of flashover. Plan immediate offline inspection.\n"
        "  Key markers: C2H2, C2H4, H2."
    ),
    "T1": (
        "Thermal fault <300 C (IEC 60599 Sec 5.3.4).\n"
        "  Action: Check overheated connections, tap-changer contacts.\n"
        "  Key marker: CH4. Re-test in 3 months."
    ),
    "T2": (
        "Thermal fault 300-700 C (IEC 60599 Sec 5.3.5).\n"
        "  Action: Hotspot in windings/core. Investigate load history.\n"
        "  Key markers: CH4, C2H4. Re-test in 1 month."
    ),
    "T3": (
        "Severe thermal fault >700 C (IEC 60599 Sec 5.3.6) — URGENT.\n"
        "  Action: Serious overheating, insulation breakdown risk. Take offline.\n"
        "  Key markers: C2H4, C2H2, CH4."
    ),
}

def parse_args():
    p = argparse.ArgumentParser(description="Transformer DGA Fault Diagnosis")
    for g in ["H2","CH4","C2H6","C2H4","C2H2"]:
        lo,hi = GAS_LIMITS[g]
        p.add_argument(f"--{g}", type=float, required=True,
                       help=f"{g} in ppm [{lo}-{hi}]")
    return p.parse_args()

def validate(args):
    gases = {"H2":args.H2,"CH4":args.CH4,"C2H6":args.C2H6,
             "C2H4":args.C2H4,"C2H2":args.C2H2}
    errs = []
    for n,v in gases.items():
        lo,hi = GAS_LIMITS[n]
        if v < lo: errs.append(f"{n} must be >= {lo} ppm (got {v})")
        elif v > hi: print(f"WARNING: {n}={v} ppm exceeds typical max ({hi} ppm).", file=sys.stderr)
    if errs:
        print("Input validation failed:", file=sys.stderr)
        for e in errs: print(f"  * {e}", file=sys.stderr)
        sys.exit(1)
    total = sum(gases.values())
    if total == 0:
        print("WARNING: All gases are 0 ppm — verify sensor data.", file=sys.stderr)
    elif total < 10:
        print(f"WARNING: Total gas={total:.1f} ppm (very low). Verify readings.", file=sys.stderr)
    return gases

def build_features(g):
    eps = 1e-6
    return np.array([[g["H2"], g["CH4"], g["C2H6"], g["C2H4"], g["C2H2"],
                      g["C2H2"]/(g["C2H4"]+eps),
                      g["CH4"] /(g["H2"]  +eps),
                      g["C2H4"]/(g["C2H6"]+eps),
                      sum(g.values())]], dtype=float)

def shap_explanation(model, X, pred_class_idx):
    try:
        import shap
        explainer = shap.TreeExplainer(model.base_model)
        sv = explainer.shap_values(X)          # list of arrays, one per class
        vals = sv[pred_class_idx][0]            # SHAP for predicted class
        ranked = sorted(enumerate(vals), key=lambda x: abs(x[1]), reverse=True)
        print("\n  TOP SHAP DRIVERS (why this diagnosis):")
        for rank, (fi, sv_val) in enumerate(ranked[:3], 1):
            direction = "increases" if sv_val > 0 else "decreases"
            fname = FEAT_NAMES[fi]
            plain = FEAT_PLAIN.get(fname, fname)
            print(f"    {rank}. {plain}")
            print(f"       -> {direction} probability of this fault "
                  f"(SHAP = {sv_val:+.3f})")
    except Exception as e:
        print(f"\n  [SHAP explanation unavailable: {e}]")

def main():
    args = parse_args()
    gases = validate(args)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    import joblib
    from src.calibration import CalibratedDGAModel

    model_path = os.path.join(project_root, "models", "xgb_dga_calibrated.joblib")
    model = joblib.load(model_path)

    result = model.predict_sample(
        gases["H2"], gases["CH4"], gases["C2H6"],
        gases["C2H4"], gases["C2H2"]
    )

    diag = result["diagnosis"]
    classes = ["Normal","PD","D1","D2","T1","T2","T3"]
    pred_idx = classes.index(diag)

    print("\n" + "="*58)
    print("  TRANSFORMER DGA FAULT DIAGNOSIS")
    print("="*58)
    print(f"  Diagnosis  : {diag}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  Reliability: {result['reliability']}")
    print("-"*58)
    print("  Class Probabilities:")
    for cls in classes:
        pct = result["probabilities"].get(cls, 0)*100
        bar = chr(9608)*int(pct/5)
        print(f"    {cls:7s} {pct:5.1f}%  {bar}")

    X = build_features(gases)
    shap_explanation(model, X, pred_idx)

    print("\n" + "-"*58)
    print("  IEC 60599 ACTION GUIDANCE:")
    print(f"  {IEC_GUIDANCE.get(diag, 'Consult IEC 60599 for guidance.')}")
    print("="*58 + "\n")

if __name__ == "__main__":
    main()
