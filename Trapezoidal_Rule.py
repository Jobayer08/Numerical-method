# Trapezoidal Rule in Python

def f(x):
    return x**2   # এখানে তোমার ফাংশন লিখবে (উদাহরণ: f(x) = x²)

# Input
a = float(input("Enter lower limit (a): "))
b = float(input("Enter upper limit (b): "))
n = int(input("Enter number of subintervals (n): "))

# Step size
h = (b - a) / n

# Apply Trapezoidal Rule
sum_y = f(a) + f(b)

for i in range(1, n):
    x = a + i * h
    sum_y += 2 * f(x)

integral = (h / 2) * sum_y

# Output
print("\nTrapezoidal Rule Result:")
print(f"Integral ≈ {integral:.6f}")
