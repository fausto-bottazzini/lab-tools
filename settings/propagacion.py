" Propagación de errores "

import numpy as np
from .matrices import jacobiano

def propagacion(f, val, cov):
    """
    Calcula la propagación de errores para una función f: ℝⁿ → ℝᵐ usando:
        Σf = J · Σx · Jᵗ

    Parámetros:
    - f: función simbólica (o lista de funciones) definidas con sympy.
         Pueden ser escalares o vectoriales.
    - val: lista de valores en los que se evalúa (uno por parámetro de entrada).
    - cov: matriz de covarianza de entrada (n x n).

    Retorna:
    - cov_f: matriz de covarianza propagada (m x m) como np.ndarray
    """
    try:
        J = jacobiano(f, val = val) # salida_Numpy=True
        cov_np = np.array(cov).astype(np.float64)
        return J @ cov_np @ J.T
    except Exception as e:
        raise RuntimeError(f"Error durante la propagación de errores: {e}")
    
def propagacion_std(f, val, stdval):
    """
    Versión simplificada de propagación cuando los errores son independientes.
    Utiliza covarianza diagonal.

    Parámetros:
    - f: función o lista de funciones simbólicas
    - val: valores donde se evalúa
    - stdval: lista de desviaciones estándar

    Retorna:
    - Lista de desviaciones estándar propagadas (√ de la diagonal de cov_f)
    """
    cov = np.diag(np.array(stdval)**2)
    cov_f = propagacion(f, val, cov)
    return np.sqrt(np.diag(cov_f)) if cov_f.shape[0] > 1 else np.sqrt(cov_f[0, 0])