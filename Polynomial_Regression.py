# polynomial_regression.py
# Mahir: Polynomial Regression (curve fitting) with degree selection, metrics, plots

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utils: build Vandermonde X
# -----------------------------
def poly_design(x, degree):
    """
    Return design matrix with columns [1, x, x^2, ..., x^degree].
    x: (n,) array-like
    degree: int >= 1
    """
    x = np.asarray(x, dtype=float).ravel()
    cols = [np.ones_like(x)]
    for p in range(1, degree + 1):
        cols.append(x ** p)
    return np.column_stack(cols)  # shape (n, degree+1)


# -----------------------------
# Metrics
# -----------------------------
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(ss_res / len(y_true)))
    return {"residuals": resid, "R2": r2, "RMSE": rmse, "SSE": ss_res, "SST": ss_tot}


# -----------------------------
# Core fit / predict
# -----------------------------
def fit_polynomial_regression(x, y, degree):
    """
    Fit y ~ β0 + β1 x + ... + β_degree x^degree using least squares.
    Returns dict containing beta, y_hat, metrics, and degree.
    """
    X = poly_design(x, degree)
    y = np.asarray(y, float).ravel()

    # Stable least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    m = regression_metrics(y, y_hat)
    return {"beta": beta, "degree": degree, "y_hat": y_hat, **m}


def predict_polynomial(beta, x):
    """
    Predict using learned beta (length degree+1 where beta[0] is intercept).
    """
    degree = len(beta) - 1
    X = poly_design(x, degree)
    return X @ np.asarray(beta, float)


# -----------------------------
# Train/Validation split
# -----------------------------
def train_valid_split(x, y, valid_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n * (1 - valid_ratio)))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    return (np.asarray(x)[tr_idx], np.asarray(y)[tr_idx],
            np.asarray(x)[va_idx], np.asarray(y)[va_idx])


# -----------------------------
# Degree selection by validation RMSE
# -----------------------------
def choose_degree(x, y, deg_min=1, deg_max=10, valid_ratio=0.25, seed=0):
    xtr, ytr, xva, yva = train_valid_split(x, y, valid_ratio, seed)
    report = []
    best = None
    for d in range(deg_min, deg_max + 1):
        model = fit_polynomial_regression(xtr, ytr, d)
        y_pred_val = predict_polynomial(model["beta"], xva)
        m_val = regression_metrics(yva, y_pred_val)
        report.append((d, m_val["RMSE"]))
        if (best is None) or (m_val["RMSE"] < best[1]):
            best = (d, m_val["RMSE"], model["beta"])
    return {"best_degree": best[0], "best_rmse": best[1], "best_beta": best[2],
            "val_report": report, "split": (xtr, ytr, xva, yva)}


# -----------------------------
# Demo
# -----------------------------
def demo():
    # ---- Synthetic data: quadratic with noise ----
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 60)
    y_true = 1.5 - 0.7*x + 0.4*x**2  # underlying curve
    y = y_true + rng.normal(0, 0.8, size=x.size)  # noisy observations

    # ---- Pick best degree ----
    sel = choose_degree(x, y, deg_min=1, deg_max=10, valid_ratio=0.3, seed=7)
    print("Vali")