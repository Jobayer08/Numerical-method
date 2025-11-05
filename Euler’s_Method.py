import math

# Differential equation define করো: dy/dx = f(x, y)
def f(x, y):
    return x + y  # উদাহরণ হিসেবে dy/dx = x + y

# প্রাথমিক শর্ত (initial conditions)
x0 = 0
y0 = 1

# ধাপের মান ও শেষ সীমা
h = 0.1      # step size
xn = 0.5     # যেখানে পর্যন্ত সমাধান চাও

# Euler’s Method Loop
x = x0
y = y0

print("x\t\ty (approx)")
print("---------------------")

while x < xn:
    y = y + h * f(x, y)
    x = x + h
    print(f"{x:.2f}\t\t{y:.6f}")

print(f"\nApproximate value at x = {xn} is y = {y:.6f}")
