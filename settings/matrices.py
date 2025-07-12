" Funciones para calcular las matrices derivadas (hessiano, jacobiano, gradiente y laplaciano) "

from .imports import*
from .derivadas import*

def hessiano(f, param = None, val = None, total = False, salida_numpy = True):
    """
    Calcula el hessiano simbólico o evaluado de una función escalar f: ℝⁿ → ℝ¹.

    Parámetros:
    - f: función escalar definida con sympy.
    - param: lista de nombres de variables respecto a las cuales derivar (por ejemplo: ["a", "b"]).
             Si None o total=True, se usan todos los parámetros de la función.
    - val: lista de valores para evaluar las derivadas (uno por cada parámetro de la función).
    - total: si True, deriva respecto a todos los parámetros sin importar 'param'.
    - salida_numpy: si True y se evalúa, devuelve un np.ndarray.

    Retorna:
    - sympy.Matrix (simbólica o evaluada) o np.ndarray (si salida_numpy=True).
    """
    p = parametros(f)

    # Determinar variables respecto de las cuales derivar
    if param is None or total:
        param_syms = p
    else:
        param_syms = [sp.symbols(v) for v in param]
        if not set(param_syms).issubset(set(p)):
            raise ValueError("Los parámetros deben ser un subconjunto de las variables de la función.")

    # Construcción del Hessiano simbólico
    expr = f(*p)
    H = sp.hessian(expr, param_syms)

    # Evaluación
    if val is not None:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(val) != len(p):
            raise ValueError("Cantidad de valores no coincide con los parámetros.")
        sustituciones = dict(zip(p, val))
        try:
            H = H.evalf(subs=sustituciones)
            return np.array(H).astype(np.float64) if salida_numpy else H
        except Exception as e:
            raise RuntimeError(f"No se pudo evaluar el Hessiano: {e}")

    return H

#-------------------------------

def jacobiano(fs, param = None, val = None, total = False, salida_numpy = True):
    """
    Calcula el jacobiano simbólico o evaluado de una lista de funciones f: R^n -> R^m.

    Parámetros:
    - fs: función o lista de funciones simbólicas
    - param: lista de nombres de variables a derivar (opcional si total=True)
    - val: valores en los que evaluar (uno por cada parámetro de la función)(opcional) 
    - total: si True, deriva respecto a todos los parámetros de cada función
    - salida_numpy: si True y val está dado, devuelve matriz NumPy evaluada

    Retorna:
    - Matriz simbólica (por defecto) o NumPy evaluada.
    """
    if not isinstance(fs, (list, tuple)):
        fs = [fs]

    ps_list = [parametros(f) for f in fs]

    # Determinar parámetros respecto de los que derivar
    if total or param is None:
        param_syms = ps_list[0]  # todos los parámetros de la primera función
    else:
        param_syms = [sp.symbols(v) for v in param]
        if not all(set(param_syms).issubset(set(p)) for p in ps_list):
            raise ValueError("Los parámetros deben ser un subconjunto de las variables de las funciones.")

    # Derivar
    derivadas = [derivadas_parciales(f) for f in fs]

    # Filtrar columnas si corresponde
    if not total and param is not None:
        jac_filtrado = []
        for f_der, p in zip(derivadas, ps_list):
            indices = [p.index(sym) for sym in param_syms]
            jac_filtrado.append([f_der[i] for i in indices])
    else:
        jac_filtrado = derivadas

    # Evaluación numérica (si corresponde)
    if val is not None:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(val) != len(ps_list[0]):
            raise ValueError("La cantidad de valores no coincide con los parámetros de las funciones.")
        try:
            jac_eval = []
            for i, fila in enumerate(jac_filtrado):
                substs = {ps_list[i][j]: val[j] for j in range(len(val))}
                jac_eval.append([expr.evalf(subs=substs) for expr in fila])
            return np.array(jac_eval, dtype=np.float64) if salida_numpy else sp.Matrix(jac_eval)
        except Exception as e:
            raise RuntimeError(f"Error al evaluar el Jacobiano: {e}")

    return sp.Matrix(jac_filtrado)

#-------------------------------

def gradiente(f, val = None, salida_numpy = True):
    """
    Calcula el gradiente (vector de derivadas parciales) de una función escalar f.

    Parámetros:
    - f: función escalar simbólica (definida con sympy).
    - val: valores en los que se evalúa (opcional).
    - salida_numpy: si True y val está dado, devuelve np.ndarray; si False, devuelve sympy.Matrix o lista.

    Retorna:
    - Lista de derivadas simbólicas, sympy.Matrix o np.ndarray evaluada.
    """
    derivadas = derivadas_parciales(f, val)

    if val is not None:
        return np.array(derivadas, dtype=np.float64) if salida_numpy else derivadas
    return sp.Matrix(derivadas)

def laplaciano(f, val = None):
    """
    Calcula el laplaciano de una función escalar f: suma de derivadas segundas ∂²f/∂xᵢ².

    Parámetros:
    - f: función escalar simbólica (definida con sympy).
    - val: valores numéricos en los que evaluar (opcional).

    Retorna:
    - Laplaciano simbólico o evaluado.
    """
    p = parametros(f)
    derivadas_2 = derivadas_parciales_segundas(f, val)
    diag = [derivadas_2[i][i] for i in range(len(p))]

    if val is not None:
        return float(np.sum(diag))  # todos los valores ya están evaluados
    return sp.simplify(sp.Add(*diag))




