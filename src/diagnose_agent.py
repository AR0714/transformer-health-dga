
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import joblib
import numpy as np
from rag_engine import RAGEngine
from agent import query_ollama, build_prompt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "models" / "xgb_dga_calibrated.joblib"

FAULT_LABELS = ["Normal","PD","D1","D2","T1","T2","T3"]

def get_gas_input():
    print("\nEnter dissolved gas concentrations (ppm):")
    gases = {}
    for gas in ["H2","CH4","C2H6","C2H4","C2H2"]:
        while True:
            try:
                val = float(input(f"  {gas}: "))
                if val < 0:
                    print("  Value must be >= 0")
                    continue
                gases[gas] = val
                break
            except ValueError:
                print("  Enter a number.")
    return gases

def run_diagnosis(model, rag, gases):
    H2,CH4,C2H6,C2H4,C2H2 = (gases[g] for g in ["H2","CH4","C2H6","C2H4","C2H2"])
    result = model.predict_sample(H2, CH4, C2H6, C2H4, C2H2)
    diagnosis   = result["diagnosis"]
    confidence  = result["confidence"]
    reliability = result["reliability"]

    print(f"\n{'='*55}")
    print(f"  DIAGNOSIS  : {diagnosis}")
    print(f"  CONFIDENCE : {confidence:.1%}  ({reliability})")
    print(f"{'='*55}")

    gas_summary = ", ".join(f"{k}={v:.0f}" for k,v in gases.items())
    query       = f"{diagnosis} fault transformer DGA"
    iec_chunks  = rag.retrieve(query, top_k=3)
    iec_context = rag.format_context(iec_chunks)

    print("\nQuerying LLM for engineering report (may take 20-40s)...")
    prompt   = build_prompt(diagnosis, confidence, gas_summary, iec_context)
    llm_resp = query_ollama(prompt)

    print("\n--- LLM Diagnostic Report ---")
    print(llm_resp)
    print("-----------------------------")

def main():
    print("Loading model and RAG engine...")
    model = joblib.load(MODEL_PATH)
    rag   = RAGEngine()
    print("Ready.\n")

    while True:
        print("\nOptions: [1] New diagnosis  [2] Quit")
        choice = input("Choice: ").strip()
        if choice == "2":
            print("Exiting.")
            break
        elif choice == "1":
            gases = get_gas_input()
            run_diagnosis(model, rag, gases)
        else:
            print("Enter 1 or 2.")

if __name__ == "__main__":
    main()
