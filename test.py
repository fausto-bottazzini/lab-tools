# prueba
from settings import DB_HOST, DB_PORT, configure_logging

configure_logging()
print(f"Conectando a {DB_HOST}:{DB_PORT}")

from settings.polyfitter import polyfitter
from settings.imports import np 
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 40)
y = x**10 + np.random.normal(0, 0.01, size=len(x))
std = np.ones_like(y) * 0.01

# Ajuste regularizado
res = polyfitter(orden=30, x_data=x, y_data=y, std=std, metodo='analitico', return_eval=True)

ejex = np.linspace(-1, 1, 100)
plt.figure()
plt.errorbar(x, y, yerr=std, fmt='o', label='Datos con error')
plt.plot(ejex, res[2](ejex), label='Ajuste polinómico', color='orange')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste polinómico de orden 10')
plt.legend()
plt.show()
