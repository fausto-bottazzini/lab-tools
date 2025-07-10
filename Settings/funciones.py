" funciones generales"

from .imports import*

def lineal(x,a,b):
  "función lineal"
  return np.array(x) * a + b

def modulo(x):
    "función módulo"
    if isinstance(x, (int, float)):
        return -x if x < 0 else x
    elif isinstance(x, list):
        return [modulo(val) for val in x]
    else:
        raise ValueError("Unsupported input type. Must be int, float, or list.")

def rad(x):
  "angulos en grados a radianes"
  return x*np.pi/180

def ang(θ):
    "Angulos en radianes a grados"
    return θ*180/np.pi

# Normalizar datos                                                        
def normalizar_por(valor, lista):
    "normaliza los elementos de una lista por un valor"
    "si se quiere normalizar un valor en vez de una lista usar [valor]"
    if not isinstance(valor, (int, float)):
        raise TypeError("`valor` must be a number.")
    return [x / valor for x in lista]

# Ordenar datos                                 #(#definir mejor)
def ordenar_por(lista, orden):
  "asocia los valores de la lista a otra a ordenar"
  return [x for _, x in sorted(zip(orden, lista), key=lambda pair: pair[0])]

# Máximos
def máximos(x, y, hdt=(0, 1, 0), grafico = False):
    "encuentra los maximos de un set de datos, puede graficarse"
    peaks, _ = find_peaks(y, height=int(hdt[0]), threshold=int(hdt[2]) ,distance=int(hdt[1]))
    xp = [x[pks] for pks in peaks] #ubicación del mxm
    yp = [y[pks] for pks in peaks] #valor del mxm
    
    if grafico:
        plt.plot(x, y, ".b", label='Datos')
        plt.plot(xp, yp, 'o', color='red', label='Picos')
        plt.axhline(hdt[0], color = "red", linewidth = 1, linestyle = "dashed", label="Hd" )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Picos en los datos')
        plt.legend()
       
    return xp, yp

# Mínimos
def mínimos(x, y, ord=1, grafico = False):
    "encuentra los minimos de un set de datos, puede graficarse"
    y=np.array(y)
    min = argrelmin(y, order=int(ord))[0]
    xm = [x[ms] for ms in min] #ubicación del min
    ym = [y[ms] for ms in min] #valor del min
    
    if grafico:
        plt.plot(x, y, ".b", label='Datos')
        plt.plot(xm, ym, 'o', color='red', label='Mínimos')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Mínimos en los datos')
        plt.legend()

    return xm, ym

def parametros(f):                                                                                
    """
    Devuelve los parámetros (variables) de f como símbolos de sympy.
    """    
    f_obj=getattr(ins.getmodule(f),f.__name__)
    p = ins.signature(f_obj).parameters
    parametros = [x for x in p]
    ps= [sp.symbols(param) for param in parametros]
    return ps

def dfstdf(df, f):
    subset = df.loc[f]
    resolucion_ch1 = subset["ResolucionVCH1"].values
    resolucion_ch2 = subset["ResolucionVCH2"].values
    std_ch1 = resolucion_ch1 / 2
    std_ch2 = resolucion_ch2 / 2
    return std_ch1, std_ch2

def dfstd(df):
    resolucion_ch1 = df["ResolucionVCH1"].values
    resolucion_ch2 = df["ResolucionVCH2"].values
    std_ch1 = resolucion_ch1 #/2
    std_ch2 = resolucion_ch2 #/2
    return std_ch1, std_ch2
