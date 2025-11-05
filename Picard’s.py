import numpy as np

def picard_solve(f, x0, y0, xn, h, max_iters=10, tol=1e-8):
    """
    Solve y' = f(x,y), y(x0)=y0 on [x0, xn] using Picard's method.
    Integral is approximated with composite trapezoidal rule on a uniform grid.
    Returns: xs (grid), ys (final iterate), iters_used
    """
    # 1) uniform grid
    N = int(round((xn - x0)/h))
    xs = x0 + np.arange(N+1)*h

    # 2) initial guess y^(0)(x) = y0 (constant function)
    y_prev = np.full(N+1, y0, dtype=float)

    for k in range(max_iters):
        # 3) compute f(x, y_prev) on grid
        fvals = f(xs, y_prev)  # must support vectorized ops or handle inside f

        # 4) cumulative integral via trapezoidal rule from x0 to each x_i
        # integral[0] = 0
        integral = np.zeros_like(xs)
        # for each segment [x_{j-1}, x_j]:
        # integral[i] = sum_{j=1..i} (h/2)*(f_{j-1}+f_j)
        integral[1:] = np.cumsum(0.5*h*(fvals[:-1] + fvals[1:]))

        # 5) next iterate
        y_next = y0 + integral

        # 6) stopping check
        err = np.max(np.abs(y_next - y_prev))
        if err < tol:
            return xs, y_next, (k+1)

        y_prev = y_next

    return xs, y_prev, max_iters
