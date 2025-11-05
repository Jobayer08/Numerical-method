def f(x0):
    return x0**3 - x0 - 2
def df(x0):
    return 3*x0**2 - 1

x0 = 1
tolerance = 1e-5
max_iterations = 100
for iteration in range(max_iterations):
    x1 = x0 - f(x0)/df(x0)
    print(f"Iteration {iteration + 1}: x = {x1:.6f}")
    if abs(x1 - x0) < tolerance:
        print(f"root found at x={x1:.6f}")
        break
    x0 = x1
else:
    print("maximum iterations reached without convergence")    