import re

# ── Intent definitions ───────────────────────────────────────────────────────
# Each intent has a list of keyword/phrase patterns (all lowercase).
# The detector picks the FIRST matching intent in priority order.

INTENTS = [
    {
        "name": "health_status",
        "patterns": [
            "okay", "ok", "fine", "healthy", "health", "status", "condition",
            "working", "good", "bad", "safe", "danger", "problem", "issue",
            "fault", "normal", "worried", "worry", "concern",
        ],
    },
    {
        "name": "voltage_query",
        "patterns": [
            "voltage", "volt", "kv", "primary", "secondary", "supply",
            "output voltage", "input voltage",
        ],
    },
    {
        "name": "gas_query",
        "patterns": [
            "gas", "gases", "h2", "hydrogen", "ch4", "methane",
            "c2h2", "acetylene", "c2h4", "ethylene", "c2h6", "ethane",
            "dga", "dissolved gas", "ppm",
        ],
    },
    {
        "name": "temperature_query",
        "patterns": [
            "temperature", "temp", "hot", "heat", "cool", "cold",
            "oil temp", "winding", "overheating", "thermal",
        ],
    },
    {
        "name": "load_query",
        "patterns": [
            "load", "loading", "capacity", "overload", "overloaded", "usage",
            "how much load", "percent load", "power",
        ],
    },
    {
        "name": "action_advice",
        "patterns": [
            "should i", "what should", "what to do", "action", "recommend",
            "advice", "suggest", "next step", "repair", "fix", "replace",
            "shutdown", "turn off", "disconnect",
        ],
    },
    {
        "name": "life_estimate",
        "patterns": [
            "life", "lifetime", "lifespan", "how long", "years left", "years", "left",
            "remaining", "age", "old", "how old", "when replace",
            "end of life", "retire",
        ],
    },
    {
        "name": "explain_fault",
        "patterns": [
            "explain", "what is", "what does", "why", "cause", "reason",
            "mean", "means", "pd", "partial discharge", "arcing",
            "d1", "d2", "t1", "t2", "t3",
        ],
    },
    {
        "name": "general_chat",
        "patterns": [],   # catch-all — always matches last
    },
]

# ── Detector ─────────────────────────────────────────────────────────────────

def detect_intent(user_text: str) -> str:
    """
    Return the intent name for a user message.
    Matching is case-insensitive, whole-word preferred.
    Falls back to 'general_chat' if nothing else matches.
    """
    text_lower = user_text.lower()

    for intent in INTENTS:
        if intent["name"] == "general_chat":
            return "general_chat"
        for pattern in intent["patterns"]:
            # Match as a word boundary to avoid partial matches
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                return intent["name"]

    return "general_chat"


def intent_label(intent_name: str) -> str:
    """Return a human-readable label for display/logging."""
    labels = {
        "health_status":    "Health Status Check",
        "voltage_query":    "Voltage Reading",
        "gas_query":        "DGA Gas Reading",
        "temperature_query":"Temperature Reading",
        "load_query":       "Load Status",
        "action_advice":    "Action / Advice",
        "life_estimate":    "Remaining Life Estimate",
        "explain_fault":    "Fault Explanation",
        "general_chat":     "General Conversation",
    }
    return labels.get(intent_name, intent_name)
