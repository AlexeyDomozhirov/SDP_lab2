import sympy as sp
import numpy as np
from scipy.optimize import minimize

# Вариант 1
x = sp.symbols('x', positive=True)
f = sp.sin(sp.log(x))

x0 = 4
a = 3
b = 8

# 1. Первая и вторая производные в точке x0 с помощью sympy
dy = sp.diff(f, x, evaluate=True, simplify=False)
d2y = sp.diff(f, x, 2, evaluate=True, simplify=False)

print(sp.N(dy, subs={x: x0}))
print(sp.N(d2y, subs={x: x0}))

# 2. Символьное представление производной
print(sp.simplify(dy))

# 3. Определённый интеграл методом прямоугольников
def rectangle_integral(func, left, right, n):
    xs = np.linspace(left, right, n + 1)
    mids = (xs[:-1] + xs[1:]) / 2
    return float(np.sum(func(mids)) * (right - left) / n)

print(rectangle_integral(lambda t: np.sin(np.log(t)), a, b, 10000))

# 4. Неопределённый интеграл с помощью sympy
F = sp.integrate(f, x)
print(sp.simplify(F))

# 5. Нелинейная оптимизация
def objective(v):
    x1, x2 = v
    return (x1 - 3) ** 2 + x2

def constraint(v):
    x1, x2 = v
    return -2 * x1 + 3 * x2 - 4

res = minimize(
    objective,
    [1.0, 2.0],
    bounds=[(0, None), (0, None)],
    constraints={'type': 'ineq', 'fun': constraint}
)

print(res.x[0])
print(res.x[1])
print(res.fun)
