# A revisar 

import numpy as np
import sympy as sp

# Interpolación polinomica de lagrange 

def interpol_lagrange(x_data, y_data, return_symbolic = False):
    """
    Devuelve una función evaluable (lambdificada) del polinomio interpolador de Lagrange 
    para los puntos dados (x_data, y_data).
    
    Parámetros:
    - x_data: lista o array de valores de x
    - y_data: lista o array de valores de y correspondientes
    - return_symbolic: si es True, también retorna el polinomio simbólico expandido
    
    Retorna:
    - f_interp: función evaluable f(x)
    - (opcional) polinomio simbólico expandido
    """
    x_data = np.array(x_data, dtype = float)
    y_data = np.array(y_data, dtype = float)
    x = sp.Symbol("x")
    n = len(x_data)

    polinomio = 0
    for i in range(n):
        numerador = 1
        denominador = 1
        for j in range(n):
            if i != j:
                numerador *= (x - x_data[j])
                denominador *= (x_data[i] - x_data[j])
        termino = (numerador / denominador) * y_data[i]
        polinomio += termino

    polinomio = sp.expand(polinomio)
    f_interp = sp.lambdify(x, polinomio, modules="numpy")

    if return_symbolic:
        return f_interp, polinomio
    return f_interp

#Interpolación Newton

def interpol_newton(x_data, y_data, return_symbolic=False):
    """
    Devuelve una función evaluable (lambdificada) del polinomio interpolador de Newton
    (diferencias divididas) para los puntos dados (x_data, y_data).
    
    Parámetros:
    - x_data: lista o array de valores de x
    - y_data: lista o array de valores de y correspondientes
    - return_symbolic: si es True, también retorna el polinomio simbólico expandido
    
    Retorna:
    - f_interp: función evaluable f(x)
    - (opcional) polinomio simbólico expandido
    """
    x_data = np.array(x_data, dtype = float)
    y_data = np.array(y_data, dtype = float)
    n = len(x_data)
    
    # Construir tabla de diferencias divididas (sólo la primera fila)
    coef = np.copy(y_data)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x_data[j:n] - x_data[0:n-j])
    
    # Construcción simbólica del polinomio
    x = sp.Symbol("x")
    polinomio = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        polinomio += term
    
    polinomio = sp.expand(polinomio)
    f_interp = sp.lambdify(x, polinomio, modules="numpy")

    if return_symbolic:
        return f_interp, polinomio
    return f_interp

#interpolacion spilines cubicos
import scipy.interpolate as si

def interpol_spline_cubico(x_data, y_data, bc_type = 'natural', return_obj = False):
    """
    Devuelve una función spline cúbica que interpola los puntos dados.
    
    Parámetros:
    - x_data, y_data: listas o arrays de los puntos (x, y)
    - bc_type: Tipo de condición de contorno ('natural', 'clamped', etc.)
    - return_obj: si es True, también devuelve el objeto spline completo

    Retorna:
    - spline_fun: función evaluable
    - (opcional) spline_obj: objeto CubicSpline de scipy
    """
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    spline_obj = si.CubicSpline(x_data, y_data, bc_type = bc_type)
    spline_fun = spline_obj.__call__

    if return_obj:
        return spline_fun, spline_obj
    return spline_fun