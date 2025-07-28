# prueba
from settings import DB_HOST, DB_PORT, configure_logging

configure_logging()
print(f"Conectando a {DB_HOST}:{DB_PORT}")

from settings.Interpolación import interpol_lagrange, interpol_newton

x_vals = [0, 1, 2]
y_vals = [1, 3, 2]

f, P = interpol_newton(x_vals, y_vals, return_symbolic=True)

print("Polinomio de Newton:", P)
print("Evaluación en x = 1.5:", f(1.5))


