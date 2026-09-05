import requests

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

def query_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 400}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama not running. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out."
    except Exception as e:
        return f"ERROR: {e}"

def build_prompt(diagnosis, confidence, gas_summary, iec_context):
    return f"""You are an expert power transformer diagnostic engineer.

DGA test result:
- Diagnosis: {diagnosis}
- Confidence: {confidence:.1%}
- Gas summary: {gas_summary}

Relevant IEC 60599 guidance:
---
{iec_context}
---

Provide:
1. Plain-English explanation of this fault (2-3 sentences).
2. Likely physical cause inside the transformer.
3. Recommended immediate actions for maintenance team.
4. IEC 60599 recommended monitoring interval.

Be concise and practical."""
