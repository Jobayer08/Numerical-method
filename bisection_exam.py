import math
def f(x):
    return math.exp(x)*(3.2*math.sin(x)-0.5*math.cos(x))
a=3
b=4
e_step=0.001
e_abs=0.001
import pandas as pd
rows = []
fa=f(a)
fb=f(b)
if fa*fb>0:
    print("The bisection method requires that f(a) and f(b) have opposite signs.")
else:
    iteration=0
    while True:
        c=(a+b)/2
        fc=f(c)
        intervel=abs(b-a)
        rows.append([iteration, a, b, c, fa, fb, fc,intervel, abs(fc)])
        if abs(fc)<e_abs and intervel/2<e_step:
            break    
        if fa*fc<0:
            b=c
            fb=fc
        else:
            a=c
            fa=fc
        iteration+=1
columns = ['Iteration', 'a', 'b', 'c', 'f(a)', 'f(b)', 'f(c)', 'Interval', '|f(c)|']
df = pd.DataFrame(rows, columns=columns)
print(df)  

root=df.iloc[-1]['c']
print(f"root found at c={root:.6f}")
print(f"f(root)={f(root):.6f}")
print(f"total iterations={len(df)}")

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(3, 4, 400)
y=np.exp(x)*(3.2*np.sin(x)-0.5*np.cos(x))
plt.plot(x, y, label='f(x)')
plt.axhline(0,color='b',linestyle="--")
plt.axvline(root,color='red',linestyle=":")
plt.legend()
plt.title('Bisection Method Result')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()