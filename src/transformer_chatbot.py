import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import requests

from sensor_simulator   import TransformerSensor
from transformer_memory import ConversationMemory
from intent_detector    import detect_intent
from rag_engine         import RAGEngine

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
MODEL_PATH   = os.path.join(_ROOT, "models", "xgb_dga_calibrated.joblib")
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# ── Feature engineering (same as training) ───────────────────────────────────
_EPS = 1e-6

def _make_features(g: dict) -> list:
    h2, ch4, c2h6, c2h4, c2h2 = (
        g["H2"], g["CH4"], g["C2H6"], g["C2H4"], g["C2H2"]
    )
    total = h2 + ch4 + c2h6 + c2h4 + c2h2
    return [
        h2, ch4, c2h6, c2h4, c2h2,
        c2h2 / (c2h4 + _EPS),
        ch4  / (h2   + _EPS),
        c2h4 / (c2h6 + _EPS),
        total,
    ]

# ── Ollama helper ─────────────────────────────────────────────────────────────
def _ask_ollama(prompt: str, temperature: float = 0.25) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 350},
            },
            timeout=120,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"

# ── Main chatbot class ────────────────────────────────────────────────────────

class TransformerChatbot:
    """
    Conversational transformer diagnostic assistant.
    Combines: TransformerSensor + XGBoost + RAG + Ollama + Memory.
    """

    def __init__(self, fault_scenario: str = "Normal"):
        # Load XGBoost model
        self.clf     = joblib.load(MODEL_PATH)
        self.rag     = RAGEngine()
        self.memory  = ConversationMemory(max_turns=8)
        self.sensor  = TransformerSensor(fault_scenario=fault_scenario)
        self._last_reading = None

    # ── Internal: run ML diagnosis on current sensor reading ─────────────────
    def _diagnose(self, reading: dict) -> dict:
        # predict_sample takes 5 positional gas args: H2, CH4, C2H6, C2H4, C2H2
        g = reading["gases"]
        result = self.clf.predict_sample(
            g["H2"], g["CH4"], g["C2H6"], g["C2H4"], g["C2H2"]
        )
        conf = result["confidence"]
        if isinstance(conf, float) and conf <= 1.0:
            conf = round(conf * 100, 1)
        else:
            conf = round(float(conf), 1)
        return {
            "diagnosis":     result["diagnosis"],
            "confidence":    conf,
            "probabilities": result.get("probabilities", {}),
        }

    # ── Intent handlers ───────────────────────────────────────────────────────
    def _handle_health_status(self, reading, diag) -> str:
        gases   = reading["gases"]
        context = self.rag.format_context(
            self.rag.retrieve(f"transformer fault {diag['diagnosis']}", top_k=2)
        )
        history = self.memory.format_for_prompt()
        prompt  = f"""You are a friendly transformer health assistant explaining to a non-expert.

Sensor readings right now:
- Load: {reading['load_pct']}%
- Oil temperature: {reading['oil_temp_c']} C
- Winding temperature: {reading['winding_temp_c']} C
- DGA gases (ppm): H2={gases['H2']:.1f}, CH4={gases['CH4']:.1f}, C2H2={gases['C2H2']:.1f}

AI Diagnosis: {diag['diagnosis']} (confidence: {diag['confidence']}%)

IEC 60599 Reference:
{context}

{f"Earlier in this conversation:{chr(10)}{history}" if history else ""}

In 3-4 plain English sentences: Is this transformer healthy? What is happening inside it?
Avoid jargon. Speak as if to a building manager, not an engineer."""
        return _ask_ollama(prompt)

    def _handle_voltage(self, reading) -> str:
        return (
            f"Right now the transformer is stepping {reading['primary_kv']} kV down to "
            f"{reading['secondary_v']} V. "
            f"That's {'normal' if abs(reading['secondary_v'] - 433) < 20 else 'slightly off'} "
            f"for a standard 433 V distribution system."
        )

    def _handle_gas(self, reading, diag) -> str:
        g = reading["gases"]
        context = self.rag.format_context(
            self.rag.retrieve(f"dissolved gas analysis {diag['diagnosis']}", top_k=2)
        )
        prompt = f"""Explain these dissolved gas levels to a non-technical person in 3 sentences:
H2={g['H2']:.1f}, CH4={g['CH4']:.1f}, C2H6={g['C2H6']:.1f}, C2H4={g['C2H4']:.1f}, C2H2={g['C2H2']:.1f} ppm
AI diagnosis: {diag['diagnosis']}
IEC reference: {context}
Keep it simple — no formulas."""
        return _ask_ollama(prompt)

    def _handle_temperature(self, reading) -> str:
        oil = reading["oil_temp_c"]
        wind = reading["winding_temp_c"]
        oil_status  = "safe" if oil < 85  else ("warning" if oil < 95  else "CRITICAL")
        wind_status = "safe" if wind < 98 else ("warning" if wind < 110 else "CRITICAL")
        return (
            f"Oil temperature is {oil:.1f} C ({oil_status} — limit is 85 C for continuous load). "
            f"Winding temperature is {wind:.1f} C ({wind_status} — limit is 98 C). "
            f"The transformer is running at {reading['load_pct']}% load, "
            f"which {'explains the heat.' if reading['load_pct'] > 70 else 'is moderate.'}"
        )

    def _handle_load(self, reading) -> str:
        pct = reading["load_pct"]
        if pct < 50:
            status = "lightly loaded — plenty of headroom."
        elif pct < 80:
            status = "moderately loaded — normal operating range."
        elif pct < 95:
            status = "heavily loaded — monitor temperatures closely."
        else:
            status = "OVERLOADED — immediate action recommended!"
        return f"Current load is {pct:.1f}%. The transformer is {status}"

    def _handle_action(self, reading, diag) -> str:
        context = self.rag.format_context(
            self.rag.retrieve(f"recommended actions {diag['diagnosis']}", top_k=3)
        )
        prompt = f"""A transformer engineer needs to advise a building manager.
Current fault: {diag['diagnosis']} (confidence {diag['confidence']}%)
Load: {reading['load_pct']}%   Oil temp: {reading['oil_temp_c']} C
IEC 60599 guidance: {context}
Give 3 specific, actionable recommendations in plain English. Number them 1, 2, 3."""
        return _ask_ollama(prompt)

    def _handle_life(self, reading, diag) -> str:
        hours = reading["operating_hours"]
        years = hours / 8760
        # Simple heuristic: Normal transformers last ~30 years; faults accelerate aging
        aging_factor = {"Normal":1.0,"PD":1.3,"D1":1.8,"D2":2.5,"T1":1.2,"T2":1.5,"T3":2.0}
        factor = aging_factor.get(diag["diagnosis"], 1.0)
        expected_total = 30.0 / factor
        remaining = max(0.0, expected_total - years)
        return (
            f"This transformer has been running for approximately {years:.1f} years "
            f"({hours:.0f} hours). "
            f"Under normal conditions it would last ~30 years. "
            f"With the current '{diag['diagnosis']}' condition (aging factor {factor}x), "
            f"the estimated remaining useful life is {remaining:.1f} years. "
            f"{'Schedule a full inspection soon.' if remaining < 5 else 'Continue regular monitoring.'}"
        )

    def _handle_explain(self, user_text: str, diag) -> str:
        context = self.rag.format_context(
            self.rag.retrieve(user_text, top_k=3)
        )
        prompt = f"""A non-technical person asked: "{user_text}"
Current transformer fault: {diag['diagnosis']}
IEC 60599 reference: {context}
Explain in 3-4 plain English sentences. No jargon."""
        return _ask_ollama(prompt)

    def _handle_general(self, user_text: str) -> str:
        history = self.memory.format_for_prompt()
        prompt = f"""You are a helpful transformer health assistant.
{f"Conversation so far:{chr(10)}{history}" if history else ""}
User: {user_text}
Reply in 2-3 friendly sentences."""
        return _ask_ollama(prompt)

    # ── Public: single-turn chat ───────────────────────────────────────────────
    def chat(self, user_text: str) -> str:
        self.memory.add_user(user_text)

        # Always get a fresh sensor reading
        reading = self.sensor.read()
        self._last_reading = reading
        diag    = self._diagnose(reading)
        intent  = detect_intent(user_text)

        if intent == "health_status":
            reply = self._handle_health_status(reading, diag)
        elif intent == "voltage_query":
            reply = self._handle_voltage(reading)
        elif intent == "gas_query":
            reply = self._handle_gas(reading, diag)
        elif intent == "temperature_query":
            reply = self._handle_temperature(reading)
        elif intent == "load_query":
            reply = self._handle_load(reading)
        elif intent == "action_advice":
            reply = self._handle_action(reading, diag)
        elif intent == "life_estimate":
            reply = self._handle_life(reading, diag)
        elif intent == "explain_fault":
            reply = self._handle_explain(user_text, diag)
        else:
            reply = self._handle_general(user_text)

        self.memory.add_assistant(reply)
        return reply

    def last_reading(self) -> dict:
        return self._last_reading
