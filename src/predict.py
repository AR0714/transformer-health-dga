# -*- coding: utf-8 -*-
# predict.py -- Transformer fault diagnosis from DGA gas readings
# Usage: python predict.py <H2> <CH4> <C2H6> <C2H4> <C2H2>
import sys, os, joblib
import numpy as np

ABSTAIN_THRESHOLD = 0.50

def main():
    if len(sys.argv) != 6:
        print("Usage: python predict.py <H2> <CH4> <C2H6> <C2H4> <C2H2>")
        sys.exit(1)
    H2, CH4, C2H6, C2H4, C2H2 = map(float, sys.argv[1:6])
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
    print("=========================\n")

if __name__ == '__main__':
    main()
