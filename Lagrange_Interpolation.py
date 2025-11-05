# Lagrange Interpolation Method

def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0.0
    
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
        
    return result


# Example data
x_points = [1, 2, 3]
y_points = [1, 8, 27]

# Interpolation point
x_interp = 2.5

# Calculate
y_interp = lagrange_interpolation(x_points, y_points, x_interp)

print(f"Interpolated value at x = {x_interp} is y = {y_interp:.4f}")
