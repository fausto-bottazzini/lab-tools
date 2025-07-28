" Ajuste polinomico"

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

def polyfitter(orden, x_data, y_data, std = None, decimales = 3, return_eval = False, return_symbolic = False, metodo = 'analitico', p0 = None):
    """
    Ajuste de mínimos cuadrados (analítico o numérico) de un polinomio de orden dado a datos con errores.
    
    Parámetros:
    ----------
    orden : int
        Orden del polinomio a ajustar.
    x_data, y_data, std : arrays o listas
        Datos experimentales y sus errores estándar.
    decimales : int
        Decimales para evaluar simbólicamente.
    return_eval : bool
        Si True, devuelve también la función evaluable lambdificada.
    return_symbolic : bool
        Si True, devuelve también el polinomio simbólico expandido.
    metodo : str
        'analitico' para solución cerrada, 'numerico' para ajuste curve_fit.
    P0 : array o lista, opcional
        Parámetros iniciales para el ajuste numérico (curve_fit).

    Retorna:
    --------
    pop : np.ndarray
        Coeficientes del polinomio ajustado.
    cov : np.ndarray
        Matriz de covarianza de los coeficientes.
    pol_eval : función (opcional)
        Función evaluable f(x).
    pol_sym : sympy.Expr (opcional)
        Polinomio simbólico ajustado expandido.
    """

    # Validaciones
    if not isinstance(orden, int) or orden < 0:
        raise ValueError("orden debe ser un entero ≥ 0")
    if orden >= len(x_data):
        raise ValueError("El orden del polinomio debe ser menor que la cantidad de datos")
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if len(x_data) != len(y_data):
        raise ValueError("x_data, y_data deben tener igual longitud")

    if std is not None:
        std = np.asarray(std, dtype=float)
        if len(std) != len(y_data):
            raise ValueError("y_data y std deben tener la misma longitud")

    # Construcción simbólica del polinomio
    coef = [sp.Symbol(f'a{i}') for i in range(orden + 1)]
    x_sym = sp.Symbol("x")
    pol_sym = sum(coef[i] * x_sym**i for i in range(orden + 1))
    f_lamb = sp.lambdify((x_sym, *coef), pol_sym, modules='numpy')

    def f(x_vals, *params):
        return f_lamb(x_vals, *params)

    if metodo not in ['analitico', 'numerico']:
        raise ValueError("método debe ser 'analitico' o 'numerico'")
    
    if metodo == 'numerico':
        if std is None:
            raise ValueError("std debe ser proporcionado para el método numérico")
        if p0 is None:
            p0 = np.ones(orden + 1)
        # Ajuste numérico
        pop, cov = curve_fit(f, x_data, y_data, sigma = std, p0 = p0, absolute_sigma = True)
    
    elif metodo == 'analitico':
    # Matriz de diseño
        A = np.vander(x_data, N = orden+1, increasing = True)  # [1, x, x^2, ..., x^o]
    # Ponderación si se da std
        if std is not None:
            W = np.diag(1 / std**2)
            AtW = A.T @ W
            H = AtW @ A
            condicion = np.linalg.cond(H)
            if condicion > 1e8:
                lamb = condicion* 1e-15
                print(f"⚠️ Advertencia: matriz mal condicionada (condición = {condicion:.2e}). Se regulariza (λ = {lamb:.2e}) .")
                H += lamb * np.eye(orden + 1)
            cov = np.linalg.inv(H)
            pop = cov @ AtW @ y_data
        else:
            H = A.T @ A
            condicion = np.linalg.cond(H)
            if condicion > 1e8:
                lamb = condicion* 1e-15
                print(f"⚠️ Advertencia: matriz mal condicionada (condición = {condicion:.2e}). Se regulariza (λ = {lamb:.2e}) .")
                H += lamb * np.eye(orden + 1)
            cov = np.linalg.inv(H)
            pop = cov @ A.T @ y_data

    salida = [pop, cov]

    if return_eval:
        f_eval = lambda x: f(x, *pop)
        salida.append(f_eval)

    if return_symbolic:
        fitted_sym = pol_sym.subs({coef[i]: pop[i] for i in range(orden + 1)}).evalf(n = decimales)
        salida.append(sp.expand(fitted_sym))


    return tuple(salida)


