# A revisar 

import numpy as np
import sympy as sp

# Interpolación polinomica de lagrange 

def InterPolLag(x_data, y_data):
    "devuelve el polinomio interpolador de un set de datos calculado en la forma de Lagrange"
    x_data = np.array(x_data, dtype = float)
    y_data = np.array(y_data, dtype = float)
    n = len(x_data)
    x = sp.Symbol("x")
    polinomio = 0 
    for i in range(0,n,1):
        numerador = 1
        denominador = 1 
        for j in range(0,n,1):
            if (i!=j):
                numerador = numerador*(x-x_data[j])
                denominador = denominador*(x_data[i]-x_data[j])
            termino = (numerador/denominador)*y_data[i]
        polinomio = polinomio + termino
    polisimple = sp.expand(polinomio)
    return sp.lambdify(x,polisimple)

#Interpolación Newton

def InterPolNew(x_data, y_data):
    "devuelve el polinomio interpolador de un set de datos calculado en la forma de Newton o diferencias finitas "
    x_data = np.array(x_data, dtype = float)
    y_data = np.array(y_data, dtype = float)
    titulo = ["i","xi","yi"]
    n = len(x_data)
    ki = np.arange(0,n,1)
    tabla = np.concatenate(([ki],[x_data],[y_data]), axis=0)
    tabla = np.transpose(tabla)
    dfinita = np.zeros(shape=(n,n), dtype = float)
    tabla = np.concatenate((tabla,dfinita), axis = 1)
    [n,m] = np.shape(tabla)
    diagonal = n-1
    j = 3
    while (j<m):
        titulo.append('F['+str(j-2)+']')
        i=0
        paso = j-2
        while(i< diagonal):
            denominador = (x_data[i+paso]-x_data[i])
            numerador = tabla[i+1,j-1]-tabla[i,j-1]
            tabla[i,j] = numerador/denominador
            i=i+1
        diagonal = diagonal-1 
        j=j+1
    dDividida = tabla[0,3:]
    n = len(dfinita)
    x = sp.Symbol("x")
    polinomio = y_data[0]
    for j in range(1,n,1):
        factor = dDividida[j-1]
        termino = 1
        for k in range(0,j,1):
            termino = termino*(x-x_data[k])
        polinomio = polinomio + termino*factor
    polisimple = polinomio.expand()
    return sp.lambdify(x,polisimple)

#interpolacion spilines cubicos
import scipy.interpolate as si
