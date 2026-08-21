def key_gas(h2, ch4, c2h6, c2h4, c2h2):
    """Return a likely fault name based on which gas is dominant. Inputs are ppm."""
    gases = {
        "PD":          h2,
        "T1/T2":       ch4,
        "T2":          c2h6,
        "T3":          c2h4,
        "D2 (arcing)": c2h2,
    }
    return max(gases, key=gases.get)