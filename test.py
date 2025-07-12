# prueba
from settings import DB_HOST, DB_PORT, configure_logging

configure_logging()
print(f"Conectando a {DB_HOST}:{DB_PORT}")


from settings.imports import * 
from settings.estetica import *
from settings.funciones import *
from settings.ajustes import *
from settings.derivadas import *

# Test de funciones
x = np.linspace(0, 10, 100)
def f(x, a, b):
    return a * np.sin(b * x) 
noise = np.random.normal(0, 0.1, len(x))
y = f(x, 2, 3) + noise
std = np.full_like(y, 0.1)  

print(derivadas_parciales(numpy_a_sympy(f)))

exit()

pop = Minimizer(f, x, y, std, [2, 3], metodo='differential_evolution', opciones={"bounds": [(0, 5), (0, 10)]})

print("Parámetros ajustados:", pop)

chip = chi2_pvalor(y, std, f(x,*pop), pop, reducido = True)
print("Chi2:", chip[0])
r2 = R2(y, f(x, *pop))
print("R2:", r2)


res = residuos(y, std ,f(x,*pop), grafico=True)
print("Residuos:", res)


# grafico 
plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=std,fmt = ".", label='Datos')
plt.plot(x, f(x, *pop), label='Ajuste', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de datos')
plt.legend()
plt.show(block = True)
