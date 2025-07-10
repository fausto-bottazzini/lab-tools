" Funciones para calcular las matrices derivadas (hessiano, jacobiano, gradiente y laplaciano) "

from .imports import*
from .derivadas import*

def hessiano(f_sp, param, val = None, total = False):    
  "calcula la matriz hessiana simbolica de una función (sympy) respecto de los parametros, con total = True se calcula sobre todas la variables, puede evaluarse"
  "param cadena de texto [a, b, c] (con "") val debe ser del tamaño de las variables aun total = True"
  p = parametros(f_sp)

  param_syms = [sp.symbols(v) for v in param]
  if not set(param_syms).issubset(set(p)):
    raise ValueError("Los parámetros a optimizar deben ser un subconjunto de las variables de la función.")

  derivadas_segundas = derivadas_parciales_segundas(f_sp, val)

  if total:
      hessian_filtered = derivadas_segundas
  else:                # Filtrar las derivadas segundas correspondientes a los parámetros a optimizar
    indices_param = [p.index(sym) for sym in param_syms]  # Índices de los parámetros relevantes
    hessian_filtered = [
      [derivadas_segundas[i][j] for j in indices_param]  # Filtrar columnas
      for i in indices_param                             # Filtrar filas
    ]

  if val is not None:
    try:
      valores = {p[i]: val[i] for i in range(len(val))}    # Crear un diccionario de sustituciones para los valores
      hessian_filtered = [[entry.evalf(subs=valores) for entry in row] for row in hessian_filtered]
      return np.array(hessian_filtered, dtype=np.float64)  # Convertir la matriz filtrada a formato numpy
    except Exception as e:
      raise TypeError(f"No se pudo evaluar el Hessiano con los valores proporcionados: {e}")
    
  return sp.Matrix(hessian_filtered)                   # Si no hay valores numéricos, devolver la matriz simbólica

#-------------------------------

def jacobiano(fs_sp, param, val=None, total=False):
    "calcula el jacobiano simbolico de una lista de funciones (sympy) respecto de los parametros, con total = True se calcula sobre todas las variables, puede evaluarse"
    if not isinstance(fs_sp, (list, tuple)):
        fs_sp = [fs_sp]  # Convertir a lista si no lo es

    param_syms = [sp.symbols(v) for v in param]
    ps_list = [parametros(f_sp) for f_sp in fs_sp] 

    if not all(set(param_syms).issubset(set(p)) for p in ps_list):
        raise ValueError("Los parámetros a optimizar deben ser un subconjunto de las variables de las funciones.")

    derivadas = [derivadas_parciales(f_sp, val=None) for f_sp in fs_sp]

    if not total:
        # Filtrar columnas correspondientes a los parámetros especificados
        jacobiano_filtered = []
        for f_der, p in zip(derivadas, ps_list):
            indices_param = [p.index(sym) for sym in param_syms]
            jacobiano_filtered.append([f_der[i] for i in indices_param])
    else:
        jacobiano_filtered = derivadas

    if val is not None:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(p) != len(val):
            raise ValueError("La cantidad de parámetros y valores no coincide")
        try:
            # Crear un diccionario de sustitución para cada función
            jacobiano_evaluado = []
            for i, f_der in enumerate(jacobiano_filtered):
                valores = {ps_list[i][j]: val[j] for j in range(len(val))}
                jacobiano_evaluado.append([entry.evalf(subs=valores) for entry in f_der])
            return np.array(jacobiano_evaluado, dtype=np.float64)  # Convertir a matriz NumPy
        except Exception as e:
            raise ValueError(f"No se pudo evaluar el Jacobiano con los valores proporcionados: {e}")

    # Si no se proporcionan valores, devolver la matriz simbólica
    return sp.Matrix(jacobiano_filtered)

#-------------------------------
def gradiente(f_sp, val = None):
   "calcula el gradiente simbolico de una función (sympy), puede evaluarse"
   gradiente = np.array(derivadas_parciales(f_sp, val))
   return gradiente

def laplaciano(f_sp, val = None):
  "calcula el laplaciano simbolico de una función (sympy), puede evaluarse"
  p = parametros(f_sp)
  ds = derivadas_parciales_segundas(f_sp)
  terminos = []
  for i in range(len(p)):
      terminos.append(ds[i][i])
  return np.sum(terminos)




