# Newton's Forward Difference Interpolation
import numpy as np

# Given data points
x = np.array([1, 2, 3, 4], float)
y = np.array([1, 8, 27, 64], float)

# Create forward difference table
n = len(y)
diff_table = np.zeros((n, n))
diff_table[:,0] = y

for j in range(1, n):
    for i in range(n-j):
        diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]

# Display table
print("Forward Difference Table:")
for i in range(n):
    print(*[f"{diff_table[i][j]:10.4f}" for j in range(n-i)])

# Interpolation point
x_interp = 2.5
h = x[1] - x[0]
p = (x_interp - x[0]) / h

# Apply Newton's forward formula
y_interp = y[0]
p_term = 1
for j in range(1, n):
    p_term *= (p - (j-1))
    y_interp += (p_term * diff_table[0][j]) / np.math.factorial(j)

print(f"\nInterpolated value at x = {x_interp} is {y_interp:.4f}")
