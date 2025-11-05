# logistic_regression_all_in_one.py
# Mahir: Logistic Regression from scratch (binary + OVR multiclass) with metrics and plots

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utils: standardization & split
# -----------------------------
def train_valid_split(X, y, valid_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n*(1-valid_ratio)))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]

def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


# -----------------------------
# Metrics
# -----------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def precision_recall_f1(y_true, y_pred):
    # binary, positive=1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


# -----------------------------
# Core: Binary Logistic Regression
# -----------------------------
class LogisticRegressionBinary:
    def __init__(self, lr=0.1, n_iter=2000, reg_lambda=0.0, fit_intercept=True, tol=1e-7, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda  # L2 penalty on weights (not bias)
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose
        self.w = None  # includes bias if fit_intercept

    @staticmethod
    def _sigmoid(z):
        # stable sigmoid
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _loss(self, Xb, y):
        """
        Binary cross-entropy with L2 on weights (excluding bias).
        """
        m = Xb.shape[0]
        z = Xb @ self.w
        p = self._sigmoid(z)
        # clip
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)
        ce = - (y*np.log(p) + (1-y)*np.log(1-p)).mean()
        if self.fit_intercept:
            w_only = self.w[1:]
        else:
            w_only = self.w
        reg = 0.5 * self.reg_lambda * np.sum(w_only**2) / m
        return ce + reg

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        Xb = self._add_intercept(X)
        m, d = Xb.shape
        self.w = np.zeros(d)

        prev_loss = np.inf
        for it in range(self.n_iter):
            z = Xb @ self.w
            p = self._sigmoid(z)
            grad = (Xb.T @ (p - y)) / m
            if self.fit_intercept:
                grad[1:] += self.reg_lambda * self.w[1:] / m
            else:
                grad += self.reg_lambda * self.w / m

            self.w -= self.lr * grad

            if self.verbose and (it % 200 == 0 or it == self.n_iter-1):
                cur_loss = self._loss(Xb, y)
                print(f"iter {it:5d}  loss={cur_loss:.6f}")
                if abs(prev_loss - cur_loss) < self.tol:
                    break
                prev_loss = cur_loss
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Xb = self._add_intercept(X)
        return self._sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)


# -----------------------------
# Multiclass (One-vs-Rest)
# -----------------------------
class LogisticRegressionOVR:
    def __init__(self, lr=0.1, n_iter=2000, reg_lambda=0.0, fit_intercept=True, tol=1e-7, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose
        self.classes_ = None
        self.models_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.models_ = []
        for c in self.classes_:
            y_bin = (y == c).astype(int)
            clf = LogisticRegressionBinary(lr=self.lr, n_iter=self.n_iter,
                                           reg_lambda=self.reg_lambda,
                                           fit_intercept=self.fit_intercept,
                                           tol=self.tol, verbose=self.verbose)
            clf.fit(X, y_bin)
            self.models_.append(clf)
        return self

    def predict_proba(self, X):
        # returns class-wise probabilities normalized (softmax-like via OvR)
        X = np.asarray(X, float)
        scores = np.column_stack([m.predict_proba(X) for m in self.models_])
        # normalize to sum=1 per row (not true softmax, but OvR heuristic)
        s = scores.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return scores / s

    def predict(self, X):
        X = np.asarray(X, float)
        scores = np.column_stack([m.predict_proba(X) for m in self.models_])
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


# -----------------------------
# 2D Demo: binary (with decision boundary)
# -----------------------------
def demo_binary_2d():
    rng = np.random.default_rng(1)
    n_per = 100
    mean0, mean1 = np.array([-1.5, -0.5]), np.array([1.2, 1.0])
    cov = np.array([[1.0, 0.3],[0.3, 1.0]])
    X0 = rng.multivariate_normal(mean0, cov, size=n_per)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per, int), np.ones(n_per, int)])

    # standardize
    mu, sigma = standardize_fit(X)
    Xs = standardize_apply(X, mu, sigma)

    # split
    Xtr, ytr, Xva, yva = train_valid_split(Xs, y, valid_ratio=0.3, seed=42)

    # train
    clf = LogisticRegressionBinary(lr=0.2, n_iter=3000, reg_lambda=1e-2, verbose=True)
    clf.fit(Xtr, ytr)

    # eval
    yhat_va = clf.predict(Xva)
    acc = accuracy(yva, yhat_va)
    prec, rec, f1 = precision_recall_f1(yva, yhat_va)
    print(f"\nValidation: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

    # plot decision boundary
    xx, yy = np.meshgrid(np.linspace(Xs[:,0].min()-2, Xs[:,0].max()+2, 200),
                         np.linspace(Xs[:,1].min()-2, Xs[:,1].max()+2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pp = clf.predict_proba(grid).reshape(xx.shape)

    plt.figure()
    cs = plt.contourf(xx, yy, pp, levels=20, alpha=0.3)
    plt.colorbar(cs, label="P(class=1)")
    plt.scatter(Xs[y==0,0], Xs[y==0,1], s=20, label="class 0")
    plt.scatter(Xs[y==1,0], Xs[y==1,1], s=20, label="class 1")
    # boundary at p=0.5
    plt.contour(xx, yy, pp, levels=[0.5], colors='k', linewidths=2)
    plt.title("Binary Logistic Regression (standardized features)")
    plt.xlabel("x1 (std)"); plt.ylabel("x2 (std)"); plt.legend(); plt.grid(True)
    plt.show()


# -----------------------------
# Multiclass Demo (OVR)
# -----------------------------
def demo_multiclass_2d():
    rng = np.random.default_rng(7)
    n = 120
    means = [np.array([-2, 0]), np.array([2, 0]), np.array([0, 2])]
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    X = np.vstack([rng.multivariate_normal(m, cov, size=n) for m in means])
    y = np.hstack([np.full(n, i) for i in range(3)])

    mu, sigma = standardize_fit(X)
    Xs = standardize_apply(X, mu, sigma)

    Xtr, ytr, Xva, yva = train_valid_split(Xs, y, valid_ratio=0.3, seed=1)

    clf = LogisticRegressionOVR(lr=0.2, n_iter=3000, reg_lambda=1e-2)
    clf.fit(Xtr, ytr)

    yhat = clf.predict(Xva)
    acc = accuracy(yva, yhat)
    print(f"\nMulticlass OVR validation accuracy: {acc:.3f}")

    # visualize regions (for 2D only)
    xx, yy = np.meshgrid(np.linspace(Xs[:,0].min()-2, Xs[:,0].max()+2, 250),
                         np.linspace(Xs[:,1].min()-2, Xs[:,1].max()+2, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = clf.predict(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, z, levels=len(np.unique(y))+1, alpha=0.25)
    for cls in np.unique(y):
        plt.scatter(Xs[y==cls,0], Xs[y==cls,1], s=15, label=f"class {cls}")
    plt.title("Logistic Regression (One-vs-Rest) decision regions")
    plt.xlabel("x1 (std)"); plt.ylabel("x2 (std)"); plt.legend(); plt.grid(True)
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Binary demo with boundary
    demo_binary_2d()

    # Multiclass demo
    demo_multiclass_2d()
