def g(x0):
    return 1/(1 + x0**2)

x0 = 0.5
tolerance = 1e-5
max_iterations = 100
for iteration in range(max_iterations):
    x1 = g(x0)
    print(f"Iteration {iteration + 1}: x = {x1:.6f}")
    if abs(x1 - x0) < tolerance:
        print(f"root found at x={x1:.6f}")
        break
    x0 = x1
else:
    print("maximum iterations reached without convergence")    