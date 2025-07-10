" derivadas y propagacón de errores"

from .imports import*
from .funciones import parametros

# Derivadas parciales (evaluadas y no evaluadas)
def derivadas_parciales(f,val=None):                                       
    "calcula las derivadas parciales simbolicas de una función (sympy), puede evaluarse"
    "utilizar funciones de sympy (np - sp)"
    p = parametros(f)
    if val is None: 
        try:
            fs = f(*p)
            return [sp.diff(fs, sym) for sym in p]
        except TypeError as e:
            if "loop of ufunc does not support argument 0 of type" in str(e):
                return "La función utiliza funciones np, se deben utilizar funciones sp"
            else:
                return "No se puede ejecutar la función debido a un TypeError"
        except Exception as e:
            return f"No se puede ejecutar la función: {e}"
    else:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(p) != len(val):
            raise ValueError("La cantidad de parámetros y valores no coincide")
        fs = f(*p)
        Df = [sp.diff(fs, sym) for sym in p]
        valores = {param: valor for param, valor in zip(p, val)}
        dfeval= [df.evalf(subs = valores) for df in Df]
        return dfeval


# Derivadas segundas
#-------------------------------
def derivar_lista(list,p):
  "deriva una lista (de objetos simbolicos) respecto de los parametros p"
  derivadas=[]
  for elemento in list:
    derivadas.append([sp.diff(elemento, x) for x in p])
  return derivadas

def derivadas_parciales_segundas(f_sp, val=None):
  "calcula las derivadas segundas simbolicas de una función (sympy), puede evaluarse"
  p = parametros(f_sp)  
  if val is None: 
    try:
      derivadas_primeras = derivadas_parciales(f_sp)
      derivadas_segundas = derivar_lista(derivadas_primeras, p)
      return derivadas_segundas
    except TypeError as e:
      if "loop of ufunc does not support argument 0 of type" in str(e):
        raise ValueError("Se detectó que la función utiliza funciones de numpy. Por favor, utilice funciones de sympy.") from e
      else:
          raise TypeError("Ha ocurrido un TypeError inesperado: {e}") from e
    except ValueError as e:
        raise ValueError(f"Error de valor en la función: {e}") from e
    except Exception as e:
      raise RuntimeError(f"No se puede ejecutar la función debido a un error inesperado: {e}") from e 
  else:
    if not isinstance(val, (list, tuple)):
      val = [val]
    if len(p) != len(val):
      raise ValueError("La cantidad de parámetros y valores no coincide")
    
    try:
      derivadas_primeras = derivadas_parciales(f_sp)
      derivadas_segundas = derivar_lista(derivadas_primeras, p)
      valores = {param: valor for param, valor in zip(p, val)}
      dfevaltot=[]
      for parcial in derivadas_segundas: 
        dfeval= [df.evalf(subs = valores) for df in parcial]
        dfevaltot.append(dfeval)
      return dfevaltot  
    except Exception as e:
      raise RuntimeError(f"Error durante la evaluación con valores proporcionados: {e}") from e

#-------------------------------