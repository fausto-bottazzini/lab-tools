# prueba
from settings import DB_HOST, DB_PORT, configure_logging

configure_logging()
print(f"Conectando a {DB_HOST}:{DB_PORT}")

from settings.polyfitter import polyfitter
from settings.imports import np

x = np.linspace(-1, 1, 20)
y = x**10 + np.random.normal(0, 0.01, size=len(x))
std = np.ones_like(y) * 0.01

# Ajuste regularizado
res = polyfitter(orden=10, x_data=x, y_data=y, std=std, metodo='analitico', regularizar=1e-8, return_eval=True)
