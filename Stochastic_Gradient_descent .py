# sgd_all_in_one.py
# Mahir: Stochastic Gradient Descent (mini-batch, momentum, nesterov, lr schedule)
# Demos: Linear regression (MSE), Logistic regression (BCE)

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utility: batching & shuffling
# -----------------------------
def iterate_minibatches(X, y, batch_size, shuffle=True, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        j = idx[start:start + batch_size]
        yield X[j], (y[j] if y is not None else None)


# -----------------------------
# Learning-rate schedules
# -----------------------------
class LRSchedule:
    def __init__(self, lr=0.1, decay_type="constant", step_every=1000, gamma=0.5):
        """
        decay_type: "constant" or "step"
        step_every: every k steps multiply lr by gamma (for 'step')
        """
        self.base_lr = lr
        self.decay_type = decay_type
        self.step_every = step_every
        self.gamma = gamma

    def lr(self, t):
        if self.decay_type == "constant":
            return self.base_lr
        elif self.decay_type == "step":
            steps = t // self.step_every
            return self.base_lr * (self.gamma ** steps)
        else:
            return self.base_lr


# -----------------------------
# SGD Optimizer (with momentum / Nesterov, weight decay)
# -----------------------------
class SGD:
    def __init__(self, lr_schedule, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.v = None  # velocity

    def step(self, w, grad, t):
        """
        w: parameters (vector)
        grad: gradient wrt w
        t: global step (int)
        """
        # L2 weight decay (exclude bias if desired outside)
        if self.weight_decay != 0.0:
            grad = grad + self.weight_decay * w

        lr = self.lr_schedule.lr(t)
        if self.v is None:
            self.v = np.zeros_like(w)

        # momentum update
        self.v = self.momentum * self.v - lr * grad

        if self.nesterov:
            w = w + self.momentum * self.v - lr * grad  # Nesterov lookahead form
        else:
            w = w + self.v
        return w


# -----------------------------
# Models & losses
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

# Linear regression with MSE
class LinearRegressionSGD:
    def __init__(self, d, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.w = np.zeros(d + (1 if fit_intercept else 0))

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def loss_and_grad(self, X, y):
        Xb = self._add_bias(X)
        y_hat = Xb @ self.w
        # MSE loss (mean 1/2 factor to simplify grad)
        residual = y_hat - y
        loss = 0.5 * np.mean(residual ** 2)
        grad = (Xb.T @ residual) / X.shape[0]
        return loss, grad

    def predict(self, X):
        Xb = self._add_bias(X)
        return Xb @ self.w


# Logistic regression (binary)
class LogisticRegressionSGD:
    def __init__(self, d, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.w = np.zeros(d + (1 if fit_intercept else 0))

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def loss_and_grad(self, X, y):
        Xb = self._add_bias(X)
        p = sigmoid(Xb @ self.w)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        # Binary cross-entropy
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        grad = (Xb.T @ (p - y)) / X.shape[0]
        return loss, grad

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# -----------------------------
# Trainer (generic)
# -----------------------------
def fit_with_sgd(model, X, y, optimizer, epochs=50, batch_size=32,
                 early_stop_patience=10, verbose=True):
    """
    model: has loss_and_grad(X, y) -> (loss, grad) and parameters model.w
    optimizer: SGD(...)
    Returns: history dict with losses
    """
    n_steps = 0
    best_loss = np.inf
    best_w = model.w.copy()
    patience_left = early_stop_patience
    history = []

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for Xb, yb in iterate_minibatches(X, y, batch_size, shuffle=True):
            loss, grad = model.loss_and_grad(Xb, yb)
            model.w = optimizer.step(model.w, grad, n_steps)
            n_steps += 1
            epoch_losses.append(loss)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        history.append(mean_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}")

        # early stopping on epoch mean loss
        if mean_loss + 1e-12 < best_loss:
            best_loss = mean_loss
            best_w = model.w.copy()
            patience_left = early_stop_patience
        else:
            patience_left -= 1
            if patience_left == 0:
                if verbose:
                    print("Early stopping triggered.")
                break

    # restore best
    model.w = best_w
    return {"loss_curve": history}


# -----------------------------
# Demo 1: Linear regression (convex)
# -----------------------------
def demo_linear():
    rng = np.random.default_rng(0)
    n, d = 400, 3
    X = rng.normal(size=(n, d))
    true_w = np.array([2.0, -1.5, 0.7])
    y = X @ true_w + 0.5 * rng.normal(size=n)

    # model
    model = LinearRegressionSGD(d=d, fit_intercept=True)
    # optimizer
    lr_sched = LRSchedule(lr=0.2, decay_type="step", step_every=300, gamma=0.5)
    opt = SGD(lr_schedule=lr_sched, momentum=0.9, nesterov=True, weight_decay=1e-4)

    hist = fit_with_sgd(model, X, y, opt, epochs=80, batch_size=32, early_stop_patience=15)

    print("\n[Linear] true weights (no bias shown):", true_w)
    print("[Linear] learned weights (bias, w):", np.round(model.w, 4))

    # plot loss
    plt.figure()
    plt.plot(hist["loss_curve"])
    plt.title("SGD Loss (Linear Regression)")
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.grid(True)


# -----------------------------
# Demo 2: Logistic regression (binary)
# -----------------------------
def demo_logistic():
    rng = np.random.default_rng(1)
    n = 600
    # two Gaussians
    X0 = rng.normal(loc=[-1.2, 0.0], scale=[1.0, 0.8], size=(n//2, 2))
    X1 = rng.normal(loc=[1.2, 0.8], scale=[1.0, 0.8], size=(n//2, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n//2, int), np.ones(n//2, int)])

    # standardize
    mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    model = LogisticRegressionSGD(d=2, fit_intercept=True)
    lr_sched = LRSchedule(lr=0.1, decay_type="step", step_every=500, gamma=0.5)
    opt = SGD(lr_schedule=lr_sched, momentum=0.9, nesterov=True, weight_decay=1e-3)

    hist = fit_with_sgd(model, Xs, y, opt, epochs=120, batch_size=64, early_stop_patience=20)

    # eval
    y_hat = model.predict(Xs)
    acc = float(np.mean(y_hat == y))
    print(f"\n[Logistic] accuracy (train set): {acc:.3f}")
    print("[Logistic] learned weights (bias, w1, w2):", np.round(model.w, 4))

    # plot loss
    plt.figure()
    plt.plot(hist["loss_curve"])
    plt.title("SGD Loss (Logistic Regression)")
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.grid(True)

    # decision boundary
    xx, yy = np.meshgrid(np.linspace(Xs[:,0].min()-2, Xs[:,0].max()+2, 200),
                         np.linspace(Xs[:,1].min()-2, Xs[:,1].max()+2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pp = model.predict_proba(grid).reshape(xx.shape)

    plt.figure()
    cs = plt.contourf(xx, yy, pp, levels=20, alpha=0.3)
    plt.colorbar(cs, label="P(class=1)")
    plt.scatter(Xs[y==0,0], Xs[y==0,1], s=12, label="class 0")
    plt.scatter(Xs[y==1,0], Xs[y==1,1], s=12, label="class 1")
    plt.contour(xx, yy, pp, levels=[0.5], colors='k', linewidths=2)
    plt.title("Logistic Regression (SGD) decision boundary")
    plt.xlabel("x1 (std)"); plt.ylabel("x2 (std)"); plt.legend(); plt.grid(True)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    demo_linear()
    demo_logistic()
    plt.show()
