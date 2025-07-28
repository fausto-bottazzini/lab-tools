# prueba
from settings import DB_HOST, DB_PORT, configure_logging

configure_logging()
print(f"Conectando a {DB_HOST}:{DB_PORT}")

from settings.Interpolación import interpol_spline_cubico
import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
y = [0, 1, 0, 1, 0]

f_spline, obj = interpol_spline_cubico(x, y, return_obj=True)

x_vals = np.linspace(0, 4, 200)
y_vals = f_spline(x_vals)

plt.plot(x, y, 'o', label='Datos')
plt.plot(x_vals, y_vals, '-', label='Spline cúbico')
plt.legend()
plt.grid()
plt.title("Interpolación por spline cúbico")
plt.show()




