# A revisar
# Extender de manera cerrada

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

def polyfitter(o, x_data, y_data, std, p0 = [], modo = 1, n = 3):
    "crea un polinomio de orden o y ajusta los datos"
    "x_data, y_data, std son listas de datos"
    "p0 son los parámetros iniciales"
    "mode (1,2,3,4) devuelve (pop, popstd), (pop, cov), (f), (polsym)"
    "f tener cuidado con orden 0, es una constante"
    "n es el número de decimales a mostrar en el modo 4"

    if modo not in [1, 2, 3, 4]:
        print("mode debe ser 1,2,3 o 4")
        modo = 1
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    std = np.array(std)    
    o = int(o)

    if p0:
        if len(p0) != o+1:
            print("p0 = o + 1")
            print("la cantidad de parámetros iniciales debe coincidir con la cantidad de coeficientes")
            return np.array(p0), np.full_like(p0, 0)
    if not p0:
        p0 = np.ones(o+1)
    if not isinstance(o, int) or o < 0:
        print("o es el orden del polinomio, un entero mayor o igual que 0")
        return np.array(p0), np.full_like(p0, 0)
    if o > 22 : # 26
        print("o<=22, proxima actualización")
        return p0, np.full_like(p0,0)
    if len(x_data) != len(y_data) or len(y_data) != len(std):
        print("x,y,std deben tener igual longitud")
        return np.array(p0), np.full_like(p0, 0)

    coef = [sp.symbols(chr(ord('a') + i)) for i in range(o+1)]
    x_sym = sp.symbols("x")
    pol = sum(coef[i]*x_sym**i for i in range(len(coef)))

    f_lambdified = sp.lambdify((x_sym, *coef), pol, modules='numpy')

    def f(x_vals, *params):
        return f_lambdified(x_vals, *params)

    pop,cov = curve_fit(f,x_data,y_data,sigma=std,p0=p0, absolute_sigma=True)
    popstd = np.sqrt(np.diag(cov))

    if modo == 1:
        return pop, popstd
    if modo == 2:
        return pop, cov 
    if modo == 3:
        return f
    if modo == 4:
        return pol.subs({coef[i]: pop[i] for i in range(len(coef))}).evalf(n=n).expand().simplify()

