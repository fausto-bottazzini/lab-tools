" Propagación de errores "

from .imports import*
from .funciones import parametros
from .derivadas import derivadas_parciales

# Propagación
def propagación(f,val,stdval):                                               
    "calcula el error del resultado de una operación a partir de propagación de errores"
    "utilizar funciones de sympy (np - sp)"
    "no utilizar np.arrays"
    if not isinstance(stdval, (list, tuple)):                                
            stdval = [stdval]
    df = derivadas_parciales(f,val)
    return sp.sqrt(sp.Add(*[(derv**2) * (std**2) for derv, std in zip(df, stdval)]))