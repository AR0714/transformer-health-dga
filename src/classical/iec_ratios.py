# iec_ratios.py
# The IEC 60599 ratio method: compare gases against each other, not one alone.

def _classify(r1, r2, r3):
    """Map the three IEC ratios to a fault code, or 'No decision'.
    r1 = C2H2/C2H4,  r2 = CH4/H2,  r3 = C2H4/C2H6."""
    if r1 is None or r2 is None or r3 is None:
        return "No decision (a gas was zero)"
    if r2 < 0.1 and r3 < 0.2:                              return "PD"
    if r1 > 1 and 0.1 <= r2 <= 0.5 and r3 > 1:             return "D1"
    if 0.6 <= r1 <= 2.5 and 0.1 <= r2 <= 1 and r3 > 2:     return "D2"
    if r1 < 0.1 and r2 > 1 and 1 <= r3 <= 4:               return "T2"
    if r1 < 0.2 and r2 > 1 and r3 > 4:                     return "T3"
    if r2 > 1 and r3 < 1:                                  return "T1"
    return "No decision (outside IEC table)"


def iec_ratios(h2, ch4, c2h6, c2h4, c2h2):
    """Compute the three IEC 60599 ratios and return them plus the diagnosed fault.
    Inputs are gas concentrations in ppm."""
    def safe(top, bottom):
        return top / bottom if bottom and bottom > 0 else None   # avoid divide-by-zero

    # if every gas is essentially zero, the ratios are just noise -> refuse to guess
    if max(h2, ch4, c2h6, c2h4, c2h2) < 1:
        return {"fault": "No decision (gases too low to interpret)"}

    r1 = safe(c2h2, c2h4)   # acetylene / ethylene
    r2 = safe(ch4, h2)      # methane   / hydrogen
    r3 = safe(c2h4, c2h6)   # ethylene  / ethane
    return {"C2H2/C2H4": r1, "CH4/H2": r2, "C2H4/C2H6": r3,
            "fault": _classify(r1, r2, r3)}