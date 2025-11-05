# Simpson’s 1/3 Rule in Python

def f(x):
    return x**2    # এখানে তোমার ফাংশন লিখবে (উদাহরণ: f(x) = x²)

# Input from user
a = float(input("Enter lower limit (a): "))
b = float(input("Enter upper limit (b): "))
n = int(input("Enter number of subintervals (n): "))

# Check: n must be even
if n % 2 != 0:
    print("Number of subintervals (n) must be even for Simpson’s 1/3 Rule.")
else:
    # Step size
    h = (b - a) / n

    # Apply Simpson’s 1/3 formula
    sum_y = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            sum_y += 2 * f(x)
        else:
            sum_y += 4 * f(x)

    integral = (h / 3) * sum_y

    # Output
    print("\nSimpson’s 1/3 Rule Result:")
    print(f"Integral ≈ {integral:.6f}")
