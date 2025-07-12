" bondad y minimizer"

# imports generales

import numpy as np                                                         # numpy
import matplotlib.pyplot as plt                                            # graficos
import scipy.stats as stats                                                # pvalor y chi2
from scipy.stats import chi2                                               # chi2
from scipy.optimize import minimize                                        # minimize (ajustes con metodos)


# Bondad (Chi^2 y p-valor)
def chi2_pvalor(y, yerr, y_mod, parametros, reducido = True):
    """
    Calcula el chi-cuadrado (χ²), el p-valor y los grados de libertad de un ajuste.

    Parámetros:
    - y: valores observados
    - yerr: errores de cada punto
    - y_mod: valores ajustados (modelo)
    - parametros: parametros optimos del modelo
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
    y = np.array(y)
    yerr = np.array(yerr)
    y_mod = np.array(y_mod)

    chi_cuadrado = np.sum(((y - y_mod) / yerr) ** 2)
    grados = len(y) - int(len(parametros))
    p_value = stats.chi2.sf(chi_cuadrado, grados)

    if reducido:
        chi_cuadrado /= grados
        return chi_cuadrado, p_value
    else:
        return chi_cuadrado, p_value, grados

def R2(y, y_mod, error = False):
    """
    Calcula el coeficiente de determinación R² (Pearson) de un ajuste.

    Parámetros:
    - y: valores observados
    - y_mod: valores ajustados (modelo)
    - error: si True, también devuelve el error estimado de R²

    Retorna:
    - R² (float)
    - Si error=True: (R², error_R²)
    
    Fórmulas:
    - R² = 1 - SS_res / SS_tot
    - Error aproximado: sqrt(4 * R² * (1 - R²) / (n - 2))
    """
    y = np.array(y)
    y_mod = np.array(y_mod)

    ss_res = np.sum((y - y_mod) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        raise ValueError("Varianza total nula: R² indefinido")

    r_squared = 1 - (ss_res / ss_tot)

    if error:
        n = len(y)
        if n <= 2:
            raise ValueError("No se puede estimar el error de R² con menos de 3 datos")
        err_r2 = np.sqrt(4 * r_squared * (1 - r_squared) / (n - 2))
        return r_squared, err_r2

    return r_squared

# residuos (cuadrados)
# histograma: #bines = np.sqrt(#med)

def residuos(y, yerr, y_mod, grafico = False, bines = None, ponderado = True):
    """
    Calcula los residuos cuadrados (normales o ponderados) de un ajuste y, opcionalmente, los grafica.

    Parámetros:
    - y: valores observados
    - yerr: errores de cada punto
    - y_mod: valores ajustados (modelo)
    - grafico: si True, grafica el histograma de residuos
    - bines: cantidad de bines del histograma (None usa raíz de N)
    - ponderado: si True, calcula residuos ponderados por el error

    Retorna:
    - residuos: residuos al cuadrado (normales o ponderados)
    """
    y = np.array(y)
    yerr = np.array(yerr)
    y_mod = np.array(y_mod)
    
    if ponderado:
        residuos = ((y - y_mod) / yerr) ** 2
    else:
        residuos = (y - y_mod) ** 2

    if grafico:
        plt.figure()
        titulo = "Histograma de residuos cuadrados"
        if ponderado:
            titulo += " ponderados"
        plt.title(titulo)
        bins = int(bines) if bines else int(np.sqrt(len(y)))
        plt.hist(residuos, bins=bins)

    return residuos

# Agregar métodos y resolver el tema de la covarainza
def Minimizer(f, x_data, y_data, std, parametros_iniciales, metodo = "curve_fit", opciones = None,
              jac_simbolico = None, hess_simbolico = None, covarianza = False):
    """
    Ajuste general con múltiples métodos.

    Parámetros:
    - f: función modelo
    - x_data, y_data: datos
    - std: errores
    - parametros_iniciales: guess inicial (polyfit, differential_evolution, dual_annealing y shgo no requieren)
    - metodo: string del método
    - opciones: opciones específicas del método
    - jac_simbolico: función que devuelve gradiente del error (exacto) 
    - hess_simbolico: función que devuelve hessiano del error (exacto)
    - covarianza: si True, devuelve también matriz de covarianza (no anda)

    Retorna:
    - params_opt: parámetros encontrados (y la covarainza en curve_fit)
    - cov: matriz de covarianza (si covarianza=True)

    Notas:
    - Métodos disponibles: "nelder-mead", "powell", "bfgs", "l-bfgs-b", "cg", "newton-cg",
      "tnc", "cobyla", "slsqp", "dogleg", "trust-constr", "trust-ncg", "trust-exact", "trust-krylov",
      "curve_fit", "polyfit", "differential_evolution", "dual_annealing", "basinhopping", "shgo".
    - Polyfit se debe especificar el grado en opciones y no devuelve covarianza.
    - Diferential Evolution, Dual Annealing y shgo requieren bounds en opciones.
    - Basinhopping requiere un método local en opciones.
    """

    metodo = metodo.lower()
    print("Método recibido:", metodo)

    def error(params):
        y_mod = f(x_data, *params)
        return np.sum(((y_data - y_mod) / std) ** 2)

    if metodo in ["nelder-mead", "powell", "bfgs", "l-bfgs-b", "cg", "newton-cg", "tnc", "cobyla", 
                  "slsqp", "dogleg", "trust-constr", "trust-ncg", "trust-exact", "trust-krylov"]:

        def jac_num(params):
            eps = 1e-6  # más grande para evitar ruido
            grad = []
            for i in range(len(params)):
                delta = np.zeros_like(params)
                delta[i] = eps
                e_plus = error(params + delta)
                e_base = error(params)
                grad.append((e_plus - e_base) / eps)
            return np.array(grad)

        def hess_num(params):
            eps = np.sqrt(np.finfo(float).eps)
            n = len(params)
            hess = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    ei = eps * np.eye(1, n, i)[0]
                    ej = eps * np.eye(1, n, j)[0]
                    hess[i, j] = (error(params + ei + ej) - error(params + ei) - error(params + ej) + error(params)) / eps**2
            return hess

        jac = jac_simbolico if jac_simbolico else (jac_num if metodo in ["newton-cg", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"] else None)
        hess = hess_simbolico if hess_simbolico else (hess_num if metodo in ["trust-ncg", "trust-krylov", "trust-exact"] else None)

        res = minimize(error, parametros_iniciales, method=metodo, jac = jac, hess = hess, options = opciones)
        params_opt = res.x

        # Calcular matriz de covarianza si se solicita
        cov = None
        if covarianza:
            try:
                if hess:
                    cov = np.linalg.inv(hess(params_opt))
                elif jac:
                    J = np.atleast_2d(jac(params_opt))
                    print("Jac shape:", J.shape)
                    print("J.T @ W @ J =", J.T @ W @ J) 
                    # Asegurar que J tenga forma (N, P)
                    if J.shape[0] != len(std) and J.shape[1] == len(std):
                        J = J.T
                    if J.shape[0] == len(std):
                        W = np.diag(1 / np.array(std) ** 2)
                        cov = np.linalg.inv(J.T @ W @ J)
                    else:
                        cov = np.full((len(params_opt), len(params_opt)), np.nan)
            except:
                cov = np.full((len(params_opt), len(params_opt)), np.nan)

        return (params_opt, cov) if covarianza else params_opt

    elif metodo == "curve_fit":
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(f, x_data, y_data, sigma=std, p0=parametros_iniciales, absolute_sigma=True, **(opciones or {}))
        return (popt, pcov) # if covarianza else popt

    elif metodo == "polyfit":
        grado = opciones.get("grado", 1) if opciones else 1
        coef = np.polyfit(x_data, y_data, deg = grado, w = 1 / np.array(std))
        # polyfit no da covarianza
        return (coef, None) if covarianza else coef

    elif metodo == "differential_evolution":
        from scipy.optimize import differential_evolution
        opciones = opciones.copy() if opciones else {}
        bounds = opciones.pop("bounds")  # eliminar para evitar duplicado
        res = differential_evolution(error, bounds = bounds, **opciones)
        return res.x

    elif metodo == "dual_annealing":
        from scipy.optimize import dual_annealing
        opciones = opciones.copy() if opciones else {}
        bounds = opciones.pop("bounds")
        res = dual_annealing(error, bounds = bounds, **opciones)
        return res.x

    elif metodo == "basinhopping":
        from scipy.optimize import basinhopping
        opciones = opciones.copy() if opciones else {}
        local_method = opciones.pop("local_method", "L-BFGS-B")
        minimizer_kwargs = {"method": local_method}
        res = basinhopping(error, parametros_iniciales, minimizer_kwargs = minimizer_kwargs, **opciones)
        return res.x

    elif metodo == "shgo":
        from scipy.optimize import shgo
        opciones = opciones.copy() if opciones else {}
        bounds = opciones.pop("bounds")
        res = shgo(error, bounds = bounds, **opciones)
        return res.x

    else:
        raise ValueError(f"Método '{metodo}' no reconocido o no implementado.")

