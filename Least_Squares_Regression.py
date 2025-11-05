# least_squares_all_in_one.py
# Mahir: Full Least Squares toolkit — simple linear, multiple/polynomial, ridge, metrics, plots, demo

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Metrics
# ---------------------------
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(ss_res / len(y_true)))
    return {"residuals": residuals, "R2": r2, "RMSE": rmse}


# ---------------------------
# 1) Simple Linear Regression (closed-form)
# ---------------------------
def simple_linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    beta1 = Sxy / Sxx
    beta0 = y_mean - beta1 * x_mean
    y_hat = beta0 + beta1 * x
    m = regression_metrics(y, y_hat)
    return {"beta0": beta0, "beta1": beta1, "y_hat": y_hat,
            "residuals": m["residuals"], "R2": m["R2"], "RMSE": m["RMSE"]}


# ---------------------------
# 2) General Least Squares (multiple regression)
#     X should include bias column if you want an intercept.
# ---------------------------
def fit_least_squares(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # stable
    y_hat = X @ beta
    m = regression_metrics(y, y_hat)
    return {"beta": beta, "y_hat": y_hat,
            "residuals": m["residuals"], "R2": m["R2"], "RMSE": m["RMSE"]}


def add_bias_column(X):
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


# ---------------------------
# 3) Polynomial Regression helper
#     Returns design matrix with [1, x, x^2, ..., x^degree]
# ---------------------------
def polynomial_design(x, degree):
    x = np.asarray(x, dtype=float)
    # features: [x, x^2, ..., x^p]
    X = np.column_stack([x ** k for k in range(1, degree + 1)])
    return add_bias_column(X)  # prepend ones for intercept


# ---------------------------
# 4) Ridge Regression (L2-regularized least squares)
#     Solve (X^T X + λ I) β = X^T y
#     X should include bias column if you want an intercept.
# ---------------------------
def ridge_fit(X, y, lam=1e-2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    y_hat = X @ beta
    m = regression_metrics(y, y_hat)
    return {"beta": beta, "y_hat": y_hat,
            "residuals": m["residuals"], "R2": m["R2"], "RMSE": m["RMSE"]}


# ---------------------------
# 5) Predict helper
# ---------------------------
def predict_linear(beta, X):
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)
    return X @ beta


# ---------------------------
# 6) Demo / Example usage
# ---------------------------
def demo():
    print("=== Simple Linear Regression Demo ===")
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([1.1, 1.9, 3.2, 3.8, 5.1], dtype=float)
    res_simple = simple_linear_regression(x, y)
    print(f"beta0 = {res_simple['beta0']:.6f}, beta1 = {res_simple['beta1']:.6f}")
    print(f"R2 = {res_simple['R2']:.6f}, RMSE = {res_simple['RMSE']:.6f}")

    # Plot data + fitted line
    xx = np.linspace(x.min(), x.max(), 200)
    yy = res_simple["beta0"] + res_simple["beta1"] * xx
    plt.figure()
    plt.scatter(x, y, label="data")
    plt.plot(xx, yy, label="fit", linewidth=2)
    plt.title("Simple Linear Regression")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()

    # Residuals
    plt.figure()
    plt.stem(x, res_simple["residuals"], use_line_collection=True)
    plt.axhline(0, color="k", linestyle="--")
    plt.title("Residuals (Simple LR)")
    plt.xlabel("x"); plt.ylabel("y - ŷ"); plt.grid(True)

    print("\n=== Polynomial Regression (degree=2) Demo ===")
    rng = np.random.default_rng(0)
    x_poly = np.linspace(0, 5, 30)
    y_true = 2 + 0.5 * x_poly - 0.2 * x_poly**2
    y_poly = y_true + rng.normal(0, 0.35, size=x_poly.size)

    X_poly = polynomial_design(x_poly, degree=2)  # columns: [1, x, x^2]
    out_poly = fit_least_squares(X_poly, y_poly)
    print("beta (β0, β1, β2):", np.round(out_poly["beta"], 6))
    print(f"R2 = {out_poly['R2']:.6f}, RMSE = {out_poly['RMSE']:.6f}")

    # Plot polynomial fit
    xx2 = np.linspace(x_poly.min(), x_poly.max(), 300)
    Xx2 = polynomial_design(xx2, degree=2)
    yy2 = predict_linear(out_poly["beta"], Xx2)

    plt.figure()
    plt.scatter(x_poly, y_poly, s=25, label="data")
    plt.plot(xx2, yy2, linewidth=2, label="poly fit (deg=2)")
    plt.plot(xx2, 2 + 0.5*xx2 - 0.2*xx2**2, linestyle="--", label="true curve")
    plt.title("Polynomial Regression (Degree 2)")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()

    print("\n=== Ridge Regression (deg=5) Demo ===")
    # High-degree poly to show regularization helps
    X_poly5 = polynomial_design(x_poly, degree=5)
    out_ridge = ridge_fit(X_poly5, y_poly, lam=1e-1)
    print("beta (ridge):", np.round(out_ridge["beta"], 4))
    print(f"R2 = {out_ridge['R2']:.6f}, RMSE = {out_ridge['RMSE']:.6f}")

    # Plot ridge fit vs ordinary (optional)
    xx5 = np.linspace(x_poly.min(), x_poly.max(), 300)
    Xx5 = polynomial_design(xx5, degree=5)
    yy5 = predict_linear(out_ridge["beta"], Xx5)

    plt.figure()
    plt.scatter(x_poly, y_poly, s=25, label="data")
    plt.plot(xx5, yy5, linewidth=2, label="ridge fit (deg=5, λ=0.1)")
    plt.title("Ridge Polynomial Regression")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()

    plt.show()


if __name__ == "__main__":
    demo()
