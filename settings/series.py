" Funciones para calcular series "

import sympy as sp
import numpy as np
import inspect as ins
import sys

def serie_fourier(f, intervalo = (-1, 1), n = 5, sistema = "trigonometrico", param = (), eval = False, num_points = 1000, simplificar = True):
    """
    Calcula la serie de Fourier simbólica de una función f definida con sympy.

    Parámetros:
    - f: función del tipo def f(x, a, ...) con variable independiente primero.
    - intervalo: tupla (a, b) de integración.
    - n: número de términos (positivos) a calcular.
    - sistema: "trigonometrico" o "exponencial".
    - param: tupla con valores de parámetros adicionales, en el orden que los recibe f.
    - eval: si True, evalúa numéricamente la serie en num_points puntos.
    - num_points: cantidad de puntos para la evaluación si eval = True.
    - simplificar: si True, aplica .simplify() a los coeficientes y a la serie final.

    Retorna:
    - serie simbólica (si eval=False)
    - o tupla (x_vals, y_vals) si eval=True

    Nota:
    - El calculo simbolico de la serie no es costoso, el evaluarlo es lento (llega a n = 15)
    """


    func_args = ins.getfullargspec(f).args  
    var_ind = sp.Symbol(func_args[0])
    param_symbols = [sp.Symbol(var) for var in func_args[1:]]  
    param_dict = dict(zip(param_symbols, param)) if param else {}

    if not func_args:
        raise ValueError("La función debe tener al menos un argumento como variable independiente.")

    L = (abs(intervalo[0]) + abs(intervalo[1])) / 2

    f_expr = f(var_ind, *param_symbols).subs(param_dict)
    if not f_expr.has(var_ind):
        raise ValueError("La función debe depender de la variable independiente.")

    if sistema == 'trigonometrico':
        def coef_a(n):
            expr =  (1 / L) * sp.integrate(f_expr * sp.cos(n * sp.pi * var_ind / L), (var_ind, intervalo[0], intervalo[1]))
            return expr.simplify() if simplificar else expr

        def coef_b(n):
            expr = (1 / L) * sp.integrate(f_expr * sp.sin(n * sp.pi * var_ind / L), (var_ind, intervalo[0], intervalo[1]))
            return expr.simplify() if simplificar else expr

        a0 = (1 / (2 * L)) * sp.integrate(f_expr, (var_ind, intervalo[0], intervalo[1]))
        serie = a0
        print("\nCalculando la serie de Fourier trigonométrica:")
        for i in range(1, n + 1):
            porcentaje = (i / n) * 100
            sys.stdout.write(f"\rProgreso: {porcentaje:.2f}%")
            sys.stdout.flush()  # Asegura que la salida se muestre inmediatamente 
            serie += coef_a(i) * sp.cos(i * sp.pi * var_ind / L) + coef_b(i) * sp.sin(i * sp.pi * var_ind / L)
        print("\nCalculo completado.")

    elif sistema == 'exponencial':
        def c_n(k):
            expr = (1 / (2 * L)) * sp.integrate(f_expr * sp.exp(-sp.I * k * sp.pi * var_ind / L), (var_ind, intervalo[0], intervalo[1]))
            return expr.simplify() if simplificar else expr

        serie = sum(c_n(k) * sp.exp(sp.I * k * sp.pi * var_ind / L) for k in range(-n, n + 1))

    else:
        raise ValueError("Sistema no reconocido. Use 'trigonometrico' o 'exponencial'.")

    if eval and param:
        serie_numerica = serie.subs(param_dict)
        a, b = float(intervalo[0]), float(intervalo[1])
        x_vals = np.linspace(a, b, num_points)
        f_lambdified = sp.lambdify(var_ind, serie_numerica.subs(sp.pi, np.pi), modules=['numpy'])
        y_vals = f_lambdified(x_vals)
        return x_vals, y_vals

    return serie.simplify() if simplificar else serie

##############

def serie_taylor(f, p0 = 0, n = 5, eval = False, param = (), points = (), simplificar = True): 
    """
    Calcula la serie de Taylor simbólica de una función f alrededor de un punto p0.

    Parámetros:
    - f: función simbólica tipo def f(x, a, ...).
    - p0: punto de desarrollo (por defecto 0).
    - n: orden de la serie (por defecto 5).
    - eval: si True, evalúa numéricamente en 'points'.
    - param: tupla con los valores de los parámetros de f.
    - points: lista o array con puntos donde evaluar.
    - simplificar: si True, aplica .simplify() a la serie final.

    Retorna:
    - Serie simbólica (si eval=False)
    - Array con valores evaluados (si eval=True)
    """

    func_args = ins.getfullargspec(f).args  
    var_ind = sp.Symbol(func_args[0])
    param_symbols = [sp.Symbol(var) for var in func_args[1:]]  
    param_dict = dict(zip(param_symbols, param)) if param else {}
    
    if not func_args:
        raise ValueError("La función debe tener al menos un argumento como variable independiente.")

    f_expr = f(var_ind, *param_symbols).subs(param_dict)
    if not f_expr.has(var_ind):
        raise ValueError("La función debe depender de la variable independiente.")
    
    # Cálculo del término constante
    taylor_series = f_expr.subs(var_ind, p0)
    print("\nCalculando la serie de Taylor:")
    for i in range(1, n + 1):
        porcentaje = (i / n) * 100
        sys.stdout.write(f"\rProgreso: {porcentaje:.2f}%")
        sys.stdout.flush() 
        deriv = f_expr.diff(var_ind, i).subs(var_ind, p0)
        term = (deriv / sp.factorial(i)) * (var_ind - p0) ** i
        taylor_series += term
    print("\nCalculo completado.")
    
    if eval:
        if len(points)==0:
            raise ValueError("Debe proporcionar los puntos donde evaluar la serie.")
        taylor_eval = taylor_series.subs(param_dict)
        f_lambdified = sp.lambdify(var_ind, taylor_eval.subs(sp.pi, np.pi), modules=['numpy'])
        y_vals = np.array([f_lambdified(pt) for pt in points])
        return y_vals


    return taylor_series.simplify() if simplificar else taylor_series