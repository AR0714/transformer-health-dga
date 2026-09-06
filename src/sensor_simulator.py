import random
import math
import time
import datetime

class TransformerSensor:
    """
    Simulates real-time sensor readings from a distribution transformer.
    Generates: voltages, load %, temperatures, operating hours, and DGA gases.
    Gas base levels are tuned per IEC 60599 fault scenario.
    """

    BASE_GASES = {
        "Normal": {"H2": 20,  "CH4": 8,   "C2H6": 4,   "C2H4": 2,   "C2H2": 0  },
        "PD":     {"H2": 800, "CH4": 20,  "C2H6": 5,   "C2H4": 3,   "C2H2": 1  },
        "D1":     {"H2": 300, "CH4": 30,  "C2H6": 10,  "C2H4": 20,  "C2H2": 50 },
        "D2":     {"H2": 500, "CH4": 50,  "C2H6": 10,  "C2H4": 200, "C2H2": 400},
        "T1":     {"H2": 60,  "CH4": 300, "C2H6": 150, "C2H4": 50,  "C2H2": 2  },
        "T2":     {"H2": 80,  "CH4": 500, "C2H6": 100, "C2H4": 300, "C2H2": 5  },
        "T3":     {"H2": 100, "CH4": 800, "C2H6": 100, "C2H4": 600, "C2H2": 10 },
    }

    def __init__(
        self,
        transformer_type="step_down",
        primary_kv=11.0,
        secondary_v=433,
        rated_mva=1.0,
        fault_scenario="Normal",
        base_hours=8760,   # hours already in service (default = 1 year)
    ):
        if fault_scenario not in self.BASE_GASES:
            raise ValueError(f"Unknown fault_scenario '{fault_scenario}'. "
                             f"Choose from: {list(self.BASE_GASES.keys())}")
        self.transformer_type = transformer_type
        self.primary_kv       = primary_kv
        self.secondary_v      = secondary_v
        self.rated_mva        = rated_mva
        self.fault_scenario   = fault_scenario
        self.base_hours       = base_hours
        self._session_start   = time.time()

    # ── internal helpers ────────────────────────────────────────────────────
    def _noise(self, value, pct=0.05):
        """Add Gaussian-like noise of +/- pct to value."""
        return value * (1.0 + random.uniform(-pct, pct))

    def _load_pct(self):
        """Sinusoidal daily load profile (peak at 18:00, trough at 06:00)."""
        hour = datetime.datetime.now().hour + datetime.datetime.now().minute / 60.0
        base = 45 + 30 * math.sin((hour - 6) * math.pi / 12)
        return round(min(100.0, max(20.0, self._noise(base, 0.10))), 1)

    # ── public API ──────────────────────────────────────────────────────────
    def read(self):
        """Return a dictionary of all sensor readings at this instant."""
        load = self._load_pct()

        # Temperatures rise with load
        oil_temp     = 25.0 + (load / 100.0) * 55.0 + self._noise(0.5, 0.05)
        winding_temp = oil_temp + 15.0 + (load / 100.0) * 10.0

        # DGA gases with 8% measurement noise
        gases = {
            gas: round(self._noise(ppm, 0.08), 2)
            for gas, ppm in self.BASE_GASES[self.fault_scenario].items()
        }

        elapsed_hours = (time.time() - self._session_start) / 3600.0

        return {
            "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
            "transformer_type": self.transformer_type,
            "rated_mva":        self.rated_mva,
            "primary_kv":       round(self._noise(self.primary_kv,  0.02), 2),
            "secondary_v":      round(self._noise(self.secondary_v, 0.02), 1),
            "load_pct":         load,
            "oil_temp_c":       round(oil_temp, 1),
            "winding_temp_c":   round(winding_temp, 1),
            "operating_hours":  round(self.base_hours + elapsed_hours, 1),
            "fault_scenario":   self.fault_scenario,
            "gases":            gases,
        }

    def read_gases_only(self):
        """Convenience method: returns only the 5 DGA gas values (ppm)."""
        return self.read()["gases"]
