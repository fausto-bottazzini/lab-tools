" bondad y minimizer"

# imports generales

import numpy as np                                                         # numpy
import matplotlib.pyplot as plt                                            # graficos
import scipy.stats as stats                                                # pvalor y chi2
from scipy.stats import chi2                                               # chi2
from scipy.optimize import curve_fit                                       # curv_fit (ajustes)
from scipy.optimize import minimize                                        # minimize (ajustes con metodos)


# Bondad (Chi^2 y p-valor)
def chi2_pvalor(y, yerr, y_mod, parametros, reducido = False):
    """
    Calcula el chi-cuadrado (χ²), el p-valor y los grados de libertad de un ajuste.

    Parámetros:
    - y: valores observados
    - yerr: errores de cada punto
    - y_mod: valores ajustados (modelo)
    - parametros: pop del modelo
    - reducido: si es True, calcula el χ² reducido (χ² / grados de libertad) y no devuelve los grados de libertad
    
    Returns:
    - chi_cuadrado: valor del estadístico χ²
    - p_value: probabilidad asociada
    - grados: grados de libertad (n - cantidad_parametros)

    Notas:
    - Error de χ²: sqrt(2 * grados)
    - χ² reducido = χ² / grados
    - Error del χ² reducido: sqrt(2 / grados)
    """
    
    cantidad_parametros = len(parametros)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    y_mod = np.asarray(y_mod)

    residuo_cuadrado_ponderado = ((y - y_mod) / yerr) ** 2
    chi_cuadrado = np.sum(residuo_cuadrado_ponderado)
    grados = len(y) - int(cantidad_parametros)
    p_value = stats.chi2.sf(chi_cuadrado, grados)

    if reducido:
        chi_cuadrado /= grados
        p_value = stats.chi2.sf(chi_cuadrado, 1)
        return chi_cuadrado, p_value
    else:
        return chi_cuadrado, p_value, grados




# Coeficiente de Pearson (R^2)
def R2(y,y_mod):
  "calcula el coeficiente de Pearson de un ajuste"
  def residuals(y, ymod):
    return y - y_mod
  ss_res2 = np.sum(residuals(y,y_mod)**2)
  ss_tot2 = np.sum((y-np.mean(y))**2)
  r_squared2 = 1 - (ss_res2 / ss_tot2)
  return r_squared2

# residuos (cuadrados)
# histograma: #bines = np.sqrt(#med)

def residuos(f, pop, x_data, y_data, std, grafico = False, bines = False):
    "calcula los residuos de un ajuste, puede graficarse"
    "#bines = np.sqrt(#med)"
    ymod = np.array(f(x_data, *pop))
    y_data = np.array(y_data)
    res = ((y_data - ymod)**2) 
    resstd = ((y_data - ymod/std)**2)
    if grafico:
        if bines:
            # plt.figure()
            # plt.title("Histograma de residuos cuadrados")
            # plt.hist(res, int(bines))
            plt.figure()
            plt.title("Histograma de residuos cuadrados ponderados")
            plt.hist(resstd, int(bines))
        else:
            # plt.figure()
            # plt.title("Histograma de residuos cuadrados")
            # plt.hist(res, int(np.sqrt(len(y_data))))
            plt.figure()
            plt.title("Histograma de residuos cuadrados ponderados")
            plt.hist(resstd, int(np.sqrt(len(y_data))))
    return res


# A corregir y mejorar
##################################################################################
Metodos = ["Nelder-Mead", "Powell", "BFGS", "L-BFGS-B", "CG", "Newton-CG", "TNC", "COBYLA", "SLSQP", "dogleg", "trust-constr", "trust-ncg", "trust-exact", "trust-krylov"] #curvefit (COBYQA)
def Minimizer(f, x_data, y_data, std, parametros_iniciales, metodo = None, opciones = None):          #usar funciones que tomen np.arrays
    "Metodos: Nelder-Mead, Powell, BFGS, L-BFGS-B, CG, Newton-CG, TNC, COBYLA, COBYQA, SLSQP, dogleg, trust-constr, trust-ncg, trust-exact, trust-krylov"
    def error(parametros):
        y_mod = f(x_data, *parametros)
        return np.sum(((y_data - y_mod)/std)**2)

    def jacobiano(parametros):
        epsilon = np.sqrt(np.finfo(float).eps)
        return np.array([(error(parametros + epsilon * np.eye(1, len(parametros), k)[0]) - error(parametros)) / epsilon for k in range(len(parametros))], dtype = float)

    def hessiano(parametros):
        epsilon = np.sqrt(np.finfo(float).eps)
        n = len(parametros)
        hess = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                ei = np.eye(1, n, i)[0] * epsilon
                ej = np.eye(1, n, j)[0] * epsilon
                hess[i, j] = (error(parametros + ei + ej) - error(parametros + ei) - error(parametros + ej) + error(parametros)) / (epsilon ** 2)
        return hess

    jac = jacobiano if metodo in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'] else None
    hess = hessiano if metodo in ['trust-ncg', 'trust-krylov', 'trust-exact'] else None
    
    resultado = minimize(error, parametros_iniciales, method=metodo, jac=jac, hess=hess, options=opciones)

    return resultado.x
#################################################################################################