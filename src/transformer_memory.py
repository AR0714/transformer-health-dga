from collections import deque
import datetime

class ConversationMemory:
    """
    Sliding-window conversation memory for the transformer chatbot.
    Keeps the last `max_turns` user+assistant exchanges.
    Each turn is stored as {"role": "user"|"assistant", "content": str, "timestamp": str}.
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._history  = deque(maxlen=max_turns * 2)  # *2 because each turn = user + assistant

    def add_user(self, text: str):
        self._history.append({
            "role":      "user",
            "content":   text.strip(),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    def add_assistant(self, text: str):
        self._history.append({
            "role":      "assistant",
            "content":   text.strip(),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    def get_history(self) -> list:
        """Return full history as a list of dicts."""
        return list(self._history)

    def format_for_prompt(self) -> str:
        """
        Format last N turns as a compact conversation block for injecting
        into an LLM prompt.  Skips the very last user turn (it will be
        added by the caller as the current question).
        """
        turns = list(self._history)
        if not turns:
            return ""
        # Exclude the most recent entry (current user message)
        prior = turns[:-1] if turns[-1]["role"] == "user" else turns
        if not prior:
            return ""
        lines = []
        for t in prior:
            label = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{label}: {t['content']}")
        return "\n".join(lines)

    def last_diagnosis(self) -> str | None:
        """Return the last assistant reply that mentions a fault class, or None."""
        fault_keywords = ["Normal", "PD", "D1", "D2", "T1", "T2", "T3",
                          "partial discharge", "arcing", "thermal"]
        for turn in reversed(list(self._history)):
            if turn["role"] == "assistant":
                for kw in fault_keywords:
                    if kw.lower() in turn["content"].lower():
                        return turn["content"]
        return None

    def clear(self):
        self._history.clear()

    def __len__(self):
        return len(self._history) // 2  # number of complete turns

    def __repr__(self):
        return f"ConversationMemory(turns={len(self)}, max={self.max_turns})"
