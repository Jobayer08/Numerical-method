# False Position Method (Regula Falsi)

def f(x):
    return x**3 - x - 2   # এখানে তোমার সমীকরণ লিখবে

def false_position(a, b, tol):
    if f(a) * f(b) > 0:
        print("No root found in this interval.")
        return None

    print("Iter\t a\t\t b\t\t x\t\t f(x)")
    iteration = 1
    while True:
        # Regula Falsi formula
        x = (a * f(b) - b * f(a)) / (f(b) - f(a))
        print(f"{iteration}\t {a:.6f}\t {b:.6f}\t {x:.6f}\t {f(x):.6f}")

        if abs(f(x)) < tol:
            print("\nRoot found at x =", round(x, 6))
            break

        # Update interval
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

        iteration += 1

# Initial guesses
a = 1
b = 2
tolerance = 0.0001

false_position(a, b, tolerance)
