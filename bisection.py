def f(x):
    return x**3 - x - 2

a=1
b=2
tolerance=1e-5
max_iterations=100

if f(a)*f(b) >= 0:
    print("The bisection method requires that f(a) and f(b) have opposite signs.")
else:
    for iteration in range(max_iterations):
        c = (a + b) / 2
        print(f"Iteration {iteration + 1}: c = {c:.6f}, f(c) = {f(c):.6f}")
        if abs(f(c)) < tolerance :
            print(f"root found at c={c:.6f}")
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    else:
        print("maximum iterations reached without convergence")