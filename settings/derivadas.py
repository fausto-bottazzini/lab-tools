" derivadas y propagacón de errores"

from .imports import*
from .funciones import parametros

def derivadas_parciales(f, val = None):
    """
    Calcula las derivadas parciales simbólicas de una función f (definida con sympy).
    Si se pasa val, evalúa las derivadas en ese punto.

    Devuelve una lista de derivadas parciales simbólicas o evaluadas.
    """
    p = parametros(f)
    
    try:
        expr = f(*p)
        derivadas = [sp.diff(expr, pi) for pi in p]
    except Exception as e:
        if "loop of ufunc does not support argument" in str(e) or "can't convert expression to float" in str(e):
            raise TypeError("La función utiliza funciones de NumPy. Reemplazá 'np' por 'sp'.")
        raise RuntimeError(f"No se pudo calcular las derivadas: {e}")
    
    if val is not None:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(val) != len(p):
            raise ValueError(f"Se esperaban {len(p)} valores (uno por parámetro), pero se recibieron {len(val)}.")
        sustituciones = dict(zip(p, val))
        derivadas = [d.evalf(subs = sustituciones) for d in derivadas]

    return derivadas


# Derivadas segundas
#-------------------------------
def derivar_lista(lista, p):
    """
    Deriva una lista de expresiones simbólicas respecto a una lista de parámetros.
    Devuelve una matriz (lista de listas) con todas las derivadas.
    """
    return [[sp.diff(expr, pi) for pi in p] for expr in lista]

def derivadas_parciales_segundas(f, val = None):
    """
    Calcula las derivadas segundas (Hessiano completo) de una función f.
    Si se pasa val, evalúa las derivadas en ese punto.
    """
    p = parametros(f)
    try:
        derivadas_1 = derivadas_parciales(f)
        derivadas_2 = derivar_lista(derivadas_1, p)
    except Exception as e:
        if "loop of ufunc does not support argument" in str(e):
            raise TypeError("La función utiliza funciones de NumPy. Reemplazá 'np' por 'sp'.")
        raise RuntimeError(f"No se pudieron calcular las derivadas segundas: {e}")

    if val is not None:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(p) != len(val):
            raise ValueError("La cantidad de valores no coincide con los parámetros de la función.")
        sustituciones = dict(zip(p, val))
        try:
            derivadas_2 = [[d.evalf(subs=sustituciones) for d in fila] for fila in derivadas_2]
        except Exception as e:
            raise RuntimeError(f"Error al evaluar las derivadas segundas: {e}")

    return derivadas_2


#-------------------------------