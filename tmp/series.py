" Funciones para calcular series "

from .imports import*

def serie_fourier(f, intervalo=(-1, 1), n=5, sistema = "trigonometrico", param=(), eval=False, num_points=1000):
    "Calcula la serie de Fourier simbolica de f_sp considerendo el intervalo, en n iteraciones (no pasarse),"
    "se puede seleccionar sistema = trigonometrico / exponencial"
    "si la funcion tiene mas parametros dar sus valores con param"
    "se puede evaluar con eval=True en num_points, devuelve x,y"
    "usar funciones sp"
    "el calculo simbolico de la serie no pesa, el evaluarlo es lento (llega a n=15)"

    func_args = inspect.getfullargspec(f).args  
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
            return (1 / L) * sp.integrate(f_expr * sp.cos(n * sp.pi * var_ind / L), (var_ind, intervalo[0], intervalo[1])).simplify()

        def coef_b(n):
            return (1 / L) * sp.integrate(f_expr * sp.sin(n * sp.pi * var_ind / L), (var_ind, intervalo[0], intervalo[1])).simplify()

        a0 = (1 / (2 * L)) * sp.integrate(f_expr, (var_ind, intervalo[0], intervalo[1]))
        serie = a0
        print("\nCalculando la serie:")
        for i in range(1, n + 1):
            porcentaje = (i / (n+1)) * 100
            sys.stdout.write(f"\rProgreso: {porcentaje:.2f}%")
            sys.stdout.flush()  # Asegura que la salida se muestre inmediatamente 
            serie += coef_a(i) * sp.cos(i * sp.pi * var_ind / L) + coef_b(i) * sp.sin(i * sp.pi * var_ind / L)
        print("\nCalculo completado.")
    elif sistema == 'exponencial':
        c_n = lambda n: (1 / (2 * L)) * sp.integrate(f_expr * sp.exp(-sp.I * n * sp.pi * var_ind / L), (var_ind, intervalo[0], intervalo[1])).simplify()
        
        serie = sum(c_n(k) * sp.exp(sp.I * k * sp.pi * var_ind / L) for k in range(-n, n + 1))

    else:
        raise ValueError("Sistema no reconocido. Use 'trigonometrico' o 'exponencial'.")

    if eval and param:
        serie_numerica = serie.subs(param_dict).simplify()
        a, b = float(intervalo[0]), float(intervalo[1])
        x_vals = np.linspace(a, b, num_points)
        f_lambdified = sp.lambdify(var_ind, serie_numerica.subs(sp.pi, np.pi), modules=['numpy'])
        y_vals = f_lambdified(x_vals)
        return x_vals, y_vals

    return serie.simplify()

##############

def serie_taylor(f, p0 = 0, n = 5, eval = False, param = (), points = ()):       # CHEQUEAR 
    """
    Calcula la serie de Taylor de una función f alrededor de p0 hasta el orden n.
    
    Parámetros:
    - f: función a aproximar.
    - p0: punto donde se centra la serie (default 0).
    - n: número de términos de la serie (default 5).
    - eval: si es True, evalúa la serie en los puntos dados.
    - param: valores de parámetros adicionales de la función f.
    - points: valores de la variable independiente donde evaluar la serie.
    
    Retorna:
    - Si eval es False, devuelve la serie simbólica.
    - Si eval es True, devuelve un array con los valores evaluados.
    """
    func_args = inspect.getfullargspec(f).args  
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
        porcentaje = (i / (n + 1)) * 100
        sys.stdout.write(f"\rProgreso: {porcentaje:.2f}%")
        sys.stdout.flush() 
        term = (f_expr.diff(var_ind, i).subs(var_ind, p0) / sp.factorial(i)) * (var_ind - p0) ** i
        taylor_series += term
    print("\nCalculo completado.")
    
    if eval:
        if len(points)==0:
            raise ValueError("Debe proporcionar los puntos donde evaluar la serie.")
        taylor_eval = taylor_series.subs(param_dict).simplify()
        f_lambdified = sp.lambdify(var_ind, taylor_eval.subs(sp.pi, np.pi), modules=['numpy'])
        y_vals = np.array([f_lambdified(pt) for pt in points])
        return y_vals

    return taylor_series.simplify()