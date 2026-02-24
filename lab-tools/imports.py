" Imports generales, necesarios para cualquier otra función. "

# imports generales

import numpy as np                                                         # numpy
import matplotlib.pyplot as plt                                            # graficos
from matplotlib.animation import FuncAnimation                             # animaciones
import scipy.stats as stats                                                # pvalor y chi2
from scipy.stats import chi2                                               # chi2
import scipy.special                                                       # funciones raras
from scipy.optimize import curve_fit                                       # curv_fit (ajustes)
from scipy.optimize import minimize                                        # minimize (ajustes con metodos)
from scipy.signal import find_peaks                                        # máximos
from scipy.signal import argrelmin                                         # mínimos
import sympy as sp                                                         # sympy 

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

# Datos = [pd.read_csv(save_folder + "abc" + str(n) + ".csv") for n in lista]

#--------------------------------

#codigo para cargar datos txt/csv de Drive (dueño de la cuenta (MyDrive))
# datos = np.loadtxt("/content/drive/MyDrive/Lab2/" + str(i)  + ".csv" ,delimiter=",", skiprows=1, encoding="latin-1")
# columna1 = datos[:, 0]

# # Ruta a la carpeta sincronizada de Google Drive en tu sistema de archivos local
# google_drive_folder = '/ruta/a/la/carpeta/sincronizada/de/Google/Drive'
# # Lista todos los archivos en la carpeta sincronizada
# for file_name in os.listdir(google_drive_folder):
#     file_path = os.path.join(google_drive_folder, file_name)
#     if os.path.isfile(file_path):
#         print(f'Archivo encontrado: {file_name}')

#-------------------------------

# para cargar a notebook
# %run 'path_to_config/common_settings.py'

##############################################################