import math

# ---- 1) Define the ODE: dy/dx = f(x,y) ----
def f(x, y):
    return x + y    # উদাহরণ: dy/dx = x + y  (ইচ্ছা করলে বদলে নাও)

# ---- 2) One-step RK4 (for bootstrap) ----
def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h,   y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# ---- 3) Milne’s Predictor–Corrector ----
def milne_solve(f, x0, y0, h, xn, tol=1e-6, max_corr_iters=5):
    """
    Returns arrays xs, ys approximating solution of y'=f(x,y), y(x0)=y0 on [x0, xn].
    Uses fixed step-size h. Bootstrap with RK4 to get y1, y2, y3 (4 points total).
    """
    # grid
    N = int(round((xn - x0) / h))
    xs = [x0 + i*h for i in range(N+1)]
    ys = [None]*(N+1)

    # Bootstrap: y0 given, y1..y3 via RK4
    ys[0] = y0
    for i in range(3):
        ys[i+1] = rk4_step(f, xs[i], ys[i], h)

    # Now apply Milne from n=3 to N-1 to get y_{n+1}
    for n in range(3, N):
        # Values used: y_{n-3}, f_{n}, f_{n-1}, f_{n-2}
        fn    = f(xs[n],   ys[n])
        fn_1  = f(xs[n-1], ys[n-1])
        fn_2  = f(xs[n-2], ys[n-2])

        # ---- Predictor ----
        y_pred = ys[n-3] + (4*h/3)*(2*fn - fn_1 + 2*fn_2)

        # ---- Corrector (iterate) ----
        y_corr = y_pred
        for _ in range(max_corr_iters):
            f_np = f(xs[n+1], y_corr)            # f_{n+1}^{(p/c)}
            y_new = ys[n-1] + (h/3)*(f_np + 4*fn + fn_1)
            if abs(y_new - y_corr) < tol:
                y_corr = y_new
                break
            y_corr = y_new

        ys[n+1] = y_corr

    return xs, ys

# ----------- Example run -----------
if __name__ == "__main__":
    x0, y0 = 0.0, 1.0
    h      = 0.1
    xn     = 0.5

    xs, ys = milne_solve(f, x0, y0, h, xn, tol=1e-8)
    print(" x\t\t y (Milne)")
    for x, y in zip(xs, ys):
        print(f"{x:.2f}\t {y:.6f}")
