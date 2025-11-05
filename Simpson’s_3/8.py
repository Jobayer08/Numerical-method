import math

# ফাংশন সংজ্ঞা (তুমি ইচ্ছামতো পরিবর্তন করতে পারো)
def f(x):
    return math.exp(x) * (3.2 * math.sin(x) - 0.5 * math.cos(x))

# সীমা এবং n সেট করো
a = 3
b = 4
n = 6   # n অবশ্যই 3-এর গুণিতক হতে হবে

# h বের করা
h = (b - a) / n

# Simpson's 3/8 Rule সূত্র প্রয়োগ
sum_val = f(a) + f(b)

for i in range(1, n):
    x = a + i * h
    if i % 3 == 0:
        sum_val += 2 * f(x)
    else:
        sum_val += 3 * f(x)

I = (3 * h / 8) * sum_val

print(f"Approximate integral = {I:.6f}")
