" funciones generales"

import numpy as np
import sympy as sp
import matplotlib as plt
from scipy.signal import find_peaks, argrelmin
import inspect as ins

def lineal(x, a, b):
    """Función lineal: y = ax + b"""
    return np.array(x) * a + b

def modulo(x):
    """Valor absoluto (módulo) para int, float, lista o array"""
    if isinstance(x, (int, float)):
        return abs(x)
    elif isinstance(x, list):
        return [modulo(val) for val in x]
    elif isinstance(x, np.ndarray):
        return np.abs(x)
    else:
        raise ValueError("Tipo de entrada no soportado. Debe ser int, float, list o np.ndarray.")

def rad(x):
    """Convierte ángulos de grados a radianes"""
    return np.array(x) * np.pi / 180

def ang(theta):
    """Convierte ángulos de radianes a grados"""
    return np.array(theta) * 180 / np.pi

def normalizar_por(valor, lista):
    """
    Normaliza los elementos de una lista o array dividiendo por un valor escalar.
    
    Parámetros:
    - valor: número escalar divisor.
    - lista: lista o array de valores numéricos.

    Retorna:
    - Lista con los elementos normalizados.

    Nota:
    - Si querés normalizar un único valor, usá [valor] como lista.
    """
    if not isinstance(valor, (int, float)):
        raise TypeError("`valor` debe ser un número.")
    return [x / valor for x in lista]

def ordenar_por(lista, orden):
    """
    Ordena `lista` según el criterio definido en `orden`.
    
    Parámetros:
    - lista: lista de valores a ordenar.
    - orden: lista con valores por los cuales se ordena `lista`.
    
    Retorna:
    - Una nueva lista de `lista`, ordenada según los valores en `orden`.

    Ejemplo:
    >>> ordenar_por(["manzana", "banana", "kiwi"], [2, 1, 3])
    ['banana', 'manzana', 'kiwi']
    """
    if len(lista) != len(orden):
        raise ValueError("`lista` y `orden` deben tener la misma longitud.")
    return [x for _, x in sorted(zip(orden, lista), key=lambda pair: pair[0])]

# Máximos
def maximos(x, y, hdt = (0, 1, 0), grafico = False):
    """
    Encuentra los máximos locales en un conjunto de datos.

    Parámetros:
    - x: eje x (lista o array)
    - y: eje y (lista o array)
    - hdt: tupla (altura mínima, distancia mínima, umbral)
    - grafico: si True, muestra gráfico con los picos detectados

    Retorna:
    - xp: posiciones de los máximos
    - yp: valores de los máximos
    """
    x = np.array(x)
    y = np.array(y)

    peaks, _ = find_peaks(y, height=hdt[0], distance=hdt[1], threshold=hdt[2])
    xp = x[peaks].tolist()
    yp = y[peaks].tolist()

    if grafico:
        plt.plot(x, y, ".b", label='Datos')
        plt.plot(xp, yp, 'o', color='red', label='Máximos')
        plt.axhline(hdt[0], color="red", linewidth=1, linestyle="dashed", label="Altura mínima")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Detección de máximos')
        plt.legend()

    return xp, yp

# Mínimos
def minimos(x, y, ord=1, grafico=False):
    """
    Encuentra los mínimos locales en un conjunto de datos.

    Parámetros:
    - x: eje x (lista o array)
    - y: eje y (lista o array)
    - ord: orden (número de puntos a cada lado para comparar)
    - grafico: si True, muestra gráfico con los mínimos detectados

    Retorna:
    - xm: posiciones de los mínimos
    - ym: valores de los mínimos
    """
    x = np.array(x)
    y = np.array(y)

    indices_min = argrelmin(y, order=int(ord))[0]
    xm = x[indices_min].tolist()
    ym = y[indices_min].tolist()

    if grafico:
        plt.plot(x, y, ".b", label='Datos')
        plt.plot(xm, ym, 'o', color='red', label='Mínimos')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Detección de mínimos')
        plt.legend()

    return xm, ym

def parametros(f):
    """
    Devuelve los parámetros simbólicos de una función f definida como: def f(x, a, ...): return ...
    """
    try:
        f_obj = getattr(ins.getmodule(f), f.__name__)
    except Exception:
        f_obj = f  # fallback si fue definida dinámicamente
    p = ins.signature(f_obj).parameters
    return [sp.symbols(k) for k in p]


def numpy_a_sympy(func):
    """
    Convierte una función definida con np.sin, np.exp, etc. a sympy.
    Solo soporta funciones simples.
    """
    source = ins.getsource(func)
    source = source.replace("np.", "sp.")
    namespace = {"sp": sp}
    exec(source, namespace)
    return namespace[func.__name__]


def dataframe_std(df, filtro = None, mitad = True):
    """
    Extrae los desvíos estándar (std) para CH1 y CH2 desde un DataFrame de pandas.

    Parámetros:
    - df: DataFrame de pandas con las columnas 'ResolucionCH1' y 'ResolucionCH2'.
    - filtro: condición booleana opcional para filtrar filas (por ejemplo: df["Canal"] == "A").
              Si no se especifica, se usa todo el DataFrame.
    - mitad: si True, divide la resolución por 2 (resolución/2).

    Retorna:
    - std_ch1: array de std para CH1
    - std_ch2: array de std para CH2
    """
    if filtro is not None:
        df = df.loc[filtro]

    if "ResolucionCH1" not in df.columns or "ResolucionCH2" not in df.columns:
        raise KeyError("El DataFrame debe contener las columnas 'ResolucionCH1' y 'ResolucionCH2'.")

    resolucion_ch1 = df["ResolucionCH1"].values
    resolucion_ch2 = df["ResolucionCH2"].values

    factor = 0.5 if mitad else 1.0
    std_ch1 = resolucion_ch1 * factor
    std_ch2 = resolucion_ch2 * factor

    return std_ch1, std_ch2