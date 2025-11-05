# Gauss Elimination Method in Python

import numpy as np

# Input: number of equations
n = int(input("Enter number of equations: "))

# Create augmented matrix [A|B]
A = np.zeros((n, n))
B = np.zeros(n)

print("\nEnter coefficients of matrix A:")
for i in range(n):
    for j in range(n):
        A[i][j] = float(input(f"A[{i+1}][{j+1}]: "))

print("\nEnter constants of matrix B:")
for i in range(n):
    B[i] = float(input(f"B[{i+1}]: "))

# Augmented matrix
aug = np.column_stack((A, B))

# Forward Elimination
for i in range(n):
    # Pivot check (avoid divide by zero)
    if aug[i][i] == 0:
        print("Mathematical Error: Zero pivot encountered.")
        break

    for j in range(i+1, n):
        ratio = aug[j][i] / aug[i][i]
        for k in range(n+1):
            aug[j][k] = aug[j][k] - ratio * aug[i][k]

# Back Substitution
x = np.zeros(n)
x[n-1] = aug[n-1][n] / aug[n-1][n-1]

for i in range(n-2, -1, -1):
    x[i] = (aug[i][n] - np.dot(aug[i][i+1:n], x[i+1:n])) / aug[i][i]

# Output results
print("\nSolution of the system:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")
