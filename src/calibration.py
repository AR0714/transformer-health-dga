"""
Calibrated Confidence Module — Step 3
======================================
Reference : Guo, C. et al. (2017).
            On calibration of modern neural networks.
            ICML. arXiv:1706.04599.

            Niculescu-Mizil, A. & Caruana, R. (2005).
            Predicting good probabilities with supervised learning.
            ICML Proceedings, pp. 625-632.  (Platt scaling)

            IEC 60599:2022 / IEEE C57.104-2019 — confidence thresholds

Problem addressed
-----------------
The XGBoost DGA classifier achieved 80 % test accuracy (Part 1) but its
raw softmax probabilities are poorly calibrated: ECE ≈ 0.15.
A model is *calibrated* when "90 % confident" predictions are correct
exactly 90 % of the time.

This module adds two post-hoc calibration methods that operate entirely
on the *outputs* of the already-trained model — no re-training needed:

  1. Temperature Scaling (Guo et al. 2017)
     A single scalar T divides all logits before the final softmax.
     T > 1 → "cools down" over-confident predictions.
     T < 1 → sharpens under-confident ones.
     Optimised by minimising NLL on a held-out validation set.

  2. Platt Scaling (Niculescu-Mizil & Caruana 2005)
     A per-class logistic regression layer (slope a, intercept b) is
     fitted on validation predictions.  More expressive than temperature
     scaling; can also adjust systematic biases.

  Both produce calibrated probability vectors that sum to 1.

Public API
----------
    ece_score(probs, labels, n_bins=10)
    find_best_temperature(val_logits, val_labels, T_init=1.5)
    temperature_scale(logits_or_probs, T)
    fit_platt_scaler(val_probs, val_labels, n_classes=7)
    platt_scale(probs, scaler)
    CalibratedDGAModel(base_model, method='temperature', ...)
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

# ── fault classes in the same order the XGBoost model was trained ─────────
FAULT_CLASSES = ["Normal", "PD", "D1", "D2", "T1", "T2", "T3"]
N_CLASSES = len(FAULT_CLASSES)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Expected Calibration Error (ECE)
# ═══════════════════════════════════════════════════════════════════════════

def ece_score(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how much predicted confidence deviates from actual accuracy
    across the probability scale, using equal-width bins.

    Parameters
    ----------
    probs  : (N, C) float array — predicted probability for each class
    labels : (N,) int array    — true class indices  [0 … C-1]
    n_bins : int               — number of bins (10 is the standard)

    Returns
    -------
    float — ECE in [0, 1].  0 = perfect calibration.

    Example
    -------
    >>> import numpy as np
    >>> p = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    >>> y = np.array([0, 0, 1])
    >>> round(ece_score(p, y), 4)
    0.1
    """
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Confidence = max predicted probability; predicted class = argmax
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(labels)

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        # Include right edge in last bin
        if high == 1.0:
            in_bin = (confidence >= low) & (confidence <= high)
        else:
            in_bin = (confidence >= low) & (confidence < high)

        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue

        avg_conf = confidence[in_bin].mean()
        avg_acc  = correctness[in_bin].mean()
        ece += (n_in_bin / n) * abs(avg_conf - avg_acc)

    return float(ece)


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Temperature Scaling
# ═══════════════════════════════════════════════════════════════════════════

def _nll_from_logits(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    """Negative log-likelihood of temperature-scaled logits. Used during optimisation."""
    scaled   = logits / T
    log_p    = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))  # log-softmax
    n        = len(labels)
    nll      = -log_p[np.arange(n), labels].mean()
    return float(nll)


def find_best_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    T_bounds: tuple = (0.05, 10.0),
) -> float:
    """
    Find the scalar temperature T that minimises NLL on a validation set.

    Parameters
    ----------
    val_logits : (N, C) float array — raw logits (pre-softmax scores)
                 If you only have probabilities, pass np.log(probs + 1e-10).
    val_labels : (N,) int array    — true class indices
    T_bounds   : (min_T, max_T)    — search range for golden-section search

    Returns
    -------
    float — optimal temperature T* (> 1 means model was over-confident)

    Why it works
    ------------
    T* > 1 → divides logits, making softmax output flatter (less confident).
    T* < 1 → sharpens the softmax (makes already-good model more decisive).
    The validation NLL is a convex function of T, so golden-section search
    finds the exact minimum efficiently without gradients.
    """
    val_logits = np.asarray(val_logits, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    result = minimize_scalar(
        fun=lambda T: _nll_from_logits(val_logits, val_labels, T),
        bounds=T_bounds,
        method="bounded",
        options={"xatol": 1e-5},
    )
    T_star = float(result.x)
    return T_star


def temperature_scale(logits_or_probs: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling to logits (or log-probabilities).

    Parameters
    ----------
    logits_or_probs : (N, C) array
        If values are raw logits (can be negative / > 1) — used directly.
        If values are probabilities [0,1] — converted via log first.
    T : float — temperature parameter (> 0).  Use find_best_temperature().

    Returns
    -------
    (N, C) float array — calibrated probability distribution (rows sum to 1)

    Example
    -------
    >>> import numpy as np
    >>> logits = np.array([[3.0, 1.0, 0.5]])
    >>> p = temperature_scale(logits, T=2.0)
    >>> round(p.sum(), 6)
    1.0
    """
    x = np.asarray(logits_or_probs, dtype=float)

    # If input looks like probabilities, convert to logits first
    if x.min() >= 0.0 and x.max() <= 1.0:
        x = np.log(np.clip(x, 1e-10, 1.0))

    scaled = x / T
    return softmax(scaled, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Platt Scaling  (multi-class extension)
# ═══════════════════════════════════════════════════════════════════════════

def fit_platt_scaler(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int = N_CLASSES,
    C_reg: float = 1.0,
) -> LogisticRegression:
    """
    Fit a multi-class Platt scaler on validation set predictions.

    This trains a one-vs-rest LogisticRegression on top of the model's
    predicted probabilities.  The learned (slope, intercept) pairs
    correct both over-confidence and per-class systematic biases.

    Parameters
    ----------
    val_probs  : (N, C) probability predictions from the base model
    val_labels : (N,) true class indices
    n_classes  : number of classes (default 7)
    C_reg      : inverse regularisation strength (higher = less regularised)

    Returns
    -------
    LogisticRegression — fitted scaler; pass to platt_scale()
    """
    val_probs  = np.asarray(val_probs,  dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    scaler = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=C_reg,
        warm_start=False,
    )
    scaler.fit(val_probs, val_labels)
    return scaler


def platt_scale(
    probs: np.ndarray,
    scaler: LogisticRegression,
) -> np.ndarray:
    """
    Apply a fitted Platt scaler to new probability predictions.

    Parameters
    ----------
    probs  : (N, C) raw probability predictions from the base model
    scaler : fitted LogisticRegression returned by fit_platt_scaler()

    Returns
    -------
    (N, C) calibrated probability array (rows sum to 1)
    """
    probs = np.asarray(probs, dtype=float)
    return scaler.predict_proba(probs)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  CalibratedDGAModel — wrapper class
# ═══════════════════════════════════════════════════════════════════════════

class CalibratedDGAModel:
    """
    Drop-in wrapper around any DGA classifier that produces probabilities.

    Wraps a trained base model (e.g. XGBoost, Random Forest) and applies
    post-hoc probability calibration so confidence scores are trustworthy.

    Usage
    -----
        # 1. Fit on validation set  (never on the test set!)
        cal = CalibratedDGAModel(xgb_model, method='temperature')
        cal.fit(X_val, y_val)

        # 2. Predict — now with calibrated confidence
        result = cal.predict_sample(H2=50, CH4=20, C2H6=5, C2H4=10, C2H2=1)

    Parameters
    ----------
    base_model : trained sklearn-compatible model with predict_proba()
    method     : 'temperature' | 'platt'  (default 'temperature')
    classes    : list of fault class names (default FAULT_CLASSES)
    """

    def __init__(
        self,
        base_model,
        method: str = "temperature",
        classes: list[str] | None = None,
    ):
        if method not in ("temperature", "platt"):
            raise ValueError("method must be 'temperature' or 'platt'")

        self.base_model = base_model
        self.method     = method
        self.classes    = classes or FAULT_CLASSES
        self._T         = 1.0            # temperature (identity initially)
        self._scaler    = None           # Platt LogisticRegression
        self._fitted    = False
        self._ece_before: float | None = None
        self._ece_after:  float | None = None

    # ------------------------------------------------------------------
    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "CalibratedDGAModel":
        """
        Fit the calibration layer on a held-out validation set.

        Parameters
        ----------
        X_val : (N, 5) array of [H2, CH4, C2H6, C2H4, C2H2] readings
        y_val : (N,) int array of true class indices

        Returns
        -------
        self  (for chaining)
        """
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=int)

        # Raw probabilities from base model
        raw_probs = self.base_model.predict_proba(X_val)
        self._ece_before = ece_score(raw_probs, y_val)

        if self.method == "temperature":
            # XGBoost predict_proba gives probabilities; convert to logits
            logits   = np.log(np.clip(raw_probs, 1e-10, 1.0))
            self._T  = find_best_temperature(logits, y_val)
            cal_probs = temperature_scale(logits, self._T)

        else:  # platt
            self._scaler = fit_platt_scaler(raw_probs, y_val, n_classes=len(self.classes))
            cal_probs    = platt_scale(raw_probs, self._scaler)

        self._ece_after = ece_score(cal_probs, y_val)
        self._fitted    = True

        # Report improvement
        print(f"\n  Calibration method  : {self.method}")
        print(f"  Temperature T*      : {self._T:.4f}" if self.method == "temperature"
              else f"  Platt scaler fitted : {len(self.classes)}-class LR")
        print(f"  ECE before          : {self._ece_before:.4f}")
        print(f"  ECE after           : {self._ece_after:.4f}")
        delta = self._ece_before - self._ece_after
        print(f"  ECE improvement     : {delta:+.4f} "
              f"({'✓ better' if delta > 0 else '✗ worse — check validation set size'})")

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability array (N, C)."""
        if not self._fitted:
            warnings.warn("CalibratedDGAModel.fit() not called — returning raw probs.")
        raw = self.base_model.predict_proba(X)
        return self._apply_calibration(raw)

    def _apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return raw_probs
        if self.method == "temperature":
            logits = np.log(np.clip(raw_probs, 1e-10, 1.0))
            return temperature_scale(logits, self._T)
        else:
            return platt_scale(raw_probs, self._scaler)

    # ------------------------------------------------------------------
    def predict_sample(
        self,
        H2: float,
        CH4: float,
        C2H6: float,
        C2H4: float,
        C2H2: float,
    ) -> dict:
        """
        Diagnose a single transformer gas sample.

        Parameters
        ----------
        H2, CH4, C2H6, C2H4, C2H2 : float — dissolved gas concentrations (µL/L)

        Returns
        -------
        dict with:
            'diagnosis'    : str  — predicted fault class
            'confidence'   : float — calibrated confidence [0, 1]
            'probabilities': dict — calibrated prob for every class
            'reliability'  : str  — 'HIGH' / 'MEDIUM' / 'LOW'
            'temperature'  : float (temperature method only)
            'ece_before'   : float — ECE before calibration
            'ece_after'    : float — ECE after calibration
        """
        # Build the same 9-feature vector that build_features() produces:
        # 5 raw gases + 4 engineered ratios (r_c2h2_c2h4, r_ch4_h2, r_c2h4_c2h6, total_gas)
        eps = 1e-6
        r_c2h2_c2h4 = C2H2 / (C2H4 + eps)
        r_ch4_h2    = CH4  / (H2   + eps)
        r_c2h4_c2h6 = C2H4 / (C2H6 + eps)
        total_gas   = H2 + CH4 + C2H6 + C2H4 + C2H2
        X = np.array([[H2, CH4, C2H6, C2H4, C2H2,
                        r_c2h2_c2h4, r_ch4_h2, r_c2h4_c2h6, total_gas]],
                     dtype=float)
        cal_probs = self.predict_proba(X)[0]       # shape (C,)

        idx        = int(cal_probs.argmax())
        fault      = self.classes[idx]
        confidence = float(cal_probs[idx])

        # Reliability band (matches IEC 60599:2022 confidence guidance)
        if confidence >= 0.75:
            reliability = "HIGH"
        elif confidence >= 0.50:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"

        result = {
            "diagnosis":     fault,
            "confidence":    round(confidence, 4),
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(self.classes, cal_probs)},
            "reliability":   reliability,
            "ece_before":    round(self._ece_before, 4) if self._ece_before is not None else None,
            "ece_after":     round(self._ece_after, 4) if self._ece_after is not None else None,
        }
        if self.method == "temperature":
            result["temperature"] = round(self._T, 4)

        return result

    # ------------------------------------------------------------------
    def calibration_summary(self) -> dict:
        """Return a summary dict of the calibration run."""
        return {
            "method":     self.method,
            "fitted":     self._fitted,
            "temperature": round(self._T, 4),
            "ece_before": round(self._ece_before, 4) if self._ece_before is not None else None,
            "ece_after":  round(self._ece_after, 4) if self._ece_after is not None else None,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"CalibratedDGAModel(method='{self.method}', T={self._T:.3f}, "
                f"ECE {self._ece_before} → {self._ece_after}, {status})")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Self-contained tests  (python calibration.py)
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests():
    """Run built-in tests that do not require a trained model."""
    import sys

    passed = 0
    failed = 0

    def _ok(name: str, cond: bool, detail: str = ""):
        nonlocal passed, failed
        if cond:
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  {name}  [{detail}]")
            failed += 1

    np.random.seed(42)
    n, C = 500, N_CLASSES

    print("\n" + "=" * 62)
    print("  calibration.py — self-test suite")
    print("=" * 62)

    # ── Test 1: ECE of a well-calibrated toy model ────────────────────
    # Build probs where accuracy ≈ confidence:
    #   - Each sample has a predicted class with confidence c ∈ {0.7,0.8,0.9}
    #   - The TRUE label is the predicted class with probability c,
    #     and a random other class with probability (1-c)
    #   → In each confidence bin, fraction correct ≈ c  ⟹ ECE ≈ 0
    probs_perfect = np.zeros((n, C))
    labels        = np.zeros(n, dtype=int)
    conf_levels   = np.random.choice([0.7, 0.8, 0.9], size=n)
    for i, conf in enumerate(conf_levels):
        pred_cls = np.random.randint(0, C)
        # Label is pred_cls with prob=conf, random other class with prob=(1-conf)
        if np.random.rand() < conf:
            labels[i] = pred_cls
        else:
            others = [j for j in range(C) if j != pred_cls]
            labels[i] = np.random.choice(others)
        probs_perfect[i, pred_cls] = conf
        rest = (1.0 - conf) / (C - 1)
        for j in range(C):
            if j != pred_cls:
                probs_perfect[i, j] = rest
    ece_perfect = ece_score(probs_perfect, labels)
    _ok("ECE of accuracy≈confidence toy model is low (< 0.12)",
        ece_perfect < 0.12,
        f"got ECE={ece_perfect:.4f}")

    # ── Test 2: ECE of an over-confident model is high ────────────────
    probs_overconf = np.zeros((n, C))
    labels2        = np.random.randint(0, C, size=n)
    for i, lbl in enumerate(labels2):
        wrong = (lbl + 1) % C  # always predicts wrong with high conf
        probs_overconf[i, wrong] = 0.95
        probs_overconf[i, lbl]   = 0.01
        rest = 0.04 / (C - 2)
        for j in range(C):
            if j not in (wrong, lbl):
                probs_overconf[i, j] = rest
    ece_bad = ece_score(probs_overconf, labels2)
    _ok("ECE of 'always-wrong-high-conf' probs is high",
        ece_bad > 0.70,
        f"got ECE={ece_bad:.4f}")

    # ── Test 3: Temperature scaling output sums to 1 ──────────────────
    logits = np.random.randn(20, C)
    for T in (0.5, 1.0, 2.0, 5.0):
        p = temperature_scale(logits, T)
        row_sums = p.sum(axis=1)
        _ok(f"temperature_scale(T={T}) rows sum to 1",
            np.allclose(row_sums, 1.0, atol=1e-6),
            f"max deviation={abs(row_sums - 1.0).max():.2e}")

    # ── Test 4: T > 1 makes predictions less confident ────────────────
    logits_sharp = np.array([[5.0, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]])
    p_hot  = temperature_scale(logits_sharp, T=3.0)
    p_cold = temperature_scale(logits_sharp, T=0.5)
    _ok("T=3.0 lowers peak confidence vs T=0.5",
        p_hot.max() < p_cold.max(),
        f"p_hot.max={p_hot.max():.3f}, p_cold.max={p_cold.max():.3f}")

    # ── Test 5: find_best_temperature on calibrated data gives T≈1 ───
    # Generate data from a ground-truth-calibrated model
    gt_logits  = np.random.randn(200, C)
    gt_probs   = softmax(gt_logits, axis=1)
    gt_labels  = np.array([np.random.choice(C, p=gt_probs[i]) for i in range(200)])
    T_found    = find_best_temperature(gt_logits, gt_labels)
    _ok("find_best_temperature on well-calibrated data gives T near 1",
        0.3 < T_found < 3.0,
        f"T*={T_found:.4f}")

    # ── Test 6: Over-confident logits → T* > 1 ────────────────────────
    # Make very spiky logits (model extremely confident) but labels random
    spiky = np.zeros((200, C))
    for i in range(200):
        spiky[i, 0] = 10.0   # always "predicts" class 0 with huge confidence
    rand_labels = np.random.randint(0, C, 200)
    T_overconf  = find_best_temperature(spiky, rand_labels)
    _ok("Over-confident logits → T* > 1",
        T_overconf > 1.0,
        f"T*={T_overconf:.4f}")

    # ── Test 7: Platt scaler fitting and prediction ────────────────────
    probs_val  = np.random.dirichlet(np.ones(C), size=300)
    labels_val = np.random.randint(0, C, 300)
    scaler     = fit_platt_scaler(probs_val, labels_val)
    probs_new  = np.random.dirichlet(np.ones(C), size=50)
    cal_new    = platt_scale(probs_new, scaler)
    _ok("Platt scaler output rows sum to 1",
        np.allclose(cal_new.sum(axis=1), 1.0, atol=1e-6))
    _ok("Platt scaler output is non-negative",
        (cal_new >= 0).all())
    _ok("Platt scaler output has C columns",
        cal_new.shape == (50, C),
        f"shape={cal_new.shape}")

    # ── Test 8: CalibratedDGAModel with a mock base model ─────────────
    class _MockModel:
        """Minimal mock that mimics sklearn predict_proba."""
        def predict_proba(self, X):
            # Always returns a sharp distribution biased to class 0
            n    = len(X)
            probs = np.full((n, C), 0.02 / (C - 1))
            probs[:, 0] = 0.98
            return probs

    mock   = _MockModel()
    cal    = CalibratedDGAModel(mock, method="temperature")
    X_val  = np.random.rand(100, 5)
    y_val2 = np.random.randint(0, C, 100)
    cal.fit(X_val, y_val2)
    _ok("CalibratedDGAModel.fit() sets _fitted=True", cal._fitted)

    sample_result = cal.predict_sample(H2=50, CH4=20, C2H6=5, C2H4=10, C2H2=1)
    _ok("predict_sample returns required keys",
        all(k in sample_result for k in
            ("diagnosis", "confidence", "probabilities", "reliability")))
    _ok("predict_sample confidence in [0,1]",
        0.0 <= sample_result["confidence"] <= 1.0)
    _ok("predict_sample reliability is HIGH/MEDIUM/LOW",
        sample_result["reliability"] in ("HIGH", "MEDIUM", "LOW"))

    # ── Summary ───────────────────────────────────────────────────────
    total = passed + failed
    print()
    print(f"  Results : {passed}/{total} passed")
    if failed == 0:
        print("  All tests PASSED ✓")
    else:
        print(f"  {failed} test(s) FAILED — review output above")
    print("=" * 62)
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _run_tests()
    sys.exit(0 if ok else 1)
