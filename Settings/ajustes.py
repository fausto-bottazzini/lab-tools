" bondad y minimizer"

# imports generales

import numpy as np                                                         # numpy
import matplotlib.pyplot as plt                                            # graficos
import scipy.stats as stats                                                # pvalor y chi2
from scipy.stats import chi2                                               # chi2
import scipy.special                                                       # funciones raras
from scipy.optimize import curve_fit                                       # curv_fit (ajustes)
from scipy.optimize import minimize                                        # minimize (ajustes con metodos)
from scipy.signal import find_peaks                                        # máximos
from scipy.signal import argrelmin                                         # mínimos
import sympy as sp                                                         # sympy 
import pandas as pd
import inspect

from scipy.optimize._numdiff import approx_derivative
from sympy import symbols, Matrix
from sympy import lambdify, hessian

import sys
import os

import math
import functools as ft                     
import inspect as ins                      
import random as rm                                                        # aleaorio                             

import statistics   

########################################################################

# Bondad (Chi^2 y p-valor)
def chi2_pvalor(x, y, yerr, y_mod, cantidad_parametros):
    "calcula el chi^2 y el p-valor de un ajuste, (devuelve tambien los grados de libertad)"
    "error Chi^2: np.sqrt(2*nu)"
    "Chi^2 Reducido: normalizar(nu, chi2)"
    "error np.sqrt(2/nu)"
    y = np.array(y)
    yerr = np.array(yerr)
    y_mod = np.array(y_mod)
    def chi2(y_mod, y, yerr):
        return np.sum(((y - y_mod) / yerr)**2)
    grados = len(y) - int(cantidad_parametros)
    chi_cuadrado = chi2(y_mod, y, yerr)
    p_value = stats.chi2.sf(chi_cuadrado, grados)
    return chi_cuadrado, p_value, grados
# np.array(


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