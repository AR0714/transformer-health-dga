# duval.py
# The Duval Triangle 1 method: turn three gases into a point and read its fault zone.
import matplotlib.pyplot as plt

def duval_percentages(ch4, c2h4, c2h2):
    """Turn the three gases into percentages of their own total."""
    s = ch4 + c2h4 + c2h2
    if s == 0:
        return None
    return 100*ch4/s, 100*c2h4/s, 100*c2h2/s

def duval_xy(p_ch4, p_c2h4, p_c2h2):
    """Turn the three percentages into an (x, y) point on the triangle."""
    return p_c2h4 + 0.5*p_ch4, 0.866*p_ch4

def duval_zone(ch4, c2h4, c2h2):
    """Return the Duval Triangle 1 fault zone for the three gases."""
    p = duval_percentages(ch4, c2h4, c2h2)
    if p is None:
        return "No gases (cannot place on triangle)"
    a, b, c = p                 # a = %CH4,  b = %C2H4,  c = %C2H2
    if a >= 98:              return "PD"
    if c >= 13 and b <= 23:  return "D1"
    if c >= 13 and b <= 40:  return "D2"
    if c < 4 and b < 20:     return "T1"
    if c < 4 and b < 50:     return "T2"
    if c < 15 and b >= 50:   return "T3"
    return "DT"              # mixture of thermal + electrical

def plot_duval(ch4, c2h4, c2h2):
    """Draw the triangle and plot this sample as a red dot, labelled with its zone."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 100, 50, 0], [0, 0, 86.6, 0], 'k-', lw=1.5)   # the triangle outline
    ax.text(-2, -4, '100% C2H2', ha='right', fontsize=9)
    ax.text(102, -4, '100% C2H4', ha='left', fontsize=9)
    ax.text(50, 89, '100% CH4', ha='center', fontsize=9)
    p = duval_percentages(ch4, c2h4, c2h2)
    zone = duval_zone(ch4, c2h4, c2h2)
    if p:
        x, y = duval_xy(*p)
        ax.plot(x, y, 'ro', markersize=11)
        ax.annotate(f'  {zone}', (x, y), color='red', fontweight='bold', fontsize=12)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Duval Triangle 1')
    plt.show()
    return zone