# ⚙️ Settings

Librería personal con configuraciones y funciones matemáticas útiles para análisis, visualización y cálculo simbólico/numérico.

## 📚 Contenido de la librería

| Módulo            | Descripción                                                                            |
|-------------------|----------------------------------------------------------------------------------------|
| `imports.py`      | Funciones y configuraciones comunes para análisis y gráficos (ante la duda agregarlo). |
| `estetica/`       | Estilos personalizados para `matplotlib`. Incluye temas "informe" y "negro".           |
| `funciones.py`    | Funciones generales variadas.                                                          |
| `ajustes.py`      | Ajustes por cuadrados mínimos: `chi2_pvalor`, `R2`, `residuos`, `Minimizer`.           |
| `derivadas.py`    | Cálculo de derivadas primeras y segundas.                                              |
| `interpolacion.py`| Interpolaciones de Lagrange, Newton y splines cúbicos.                                 |
| `matrices.py`     | Matrices de derivadas: gradiente, jacobiano, hessiano, laplaciano.                     |
| `polyfitter.py`   | Ajuste polinomial por cuadrados mínimos (numérico y analítico).                        |
| `propagacion.py`  | Propagación de incertidumbre mediante matriz de covarianza.                            |
| `series.py`       | Series de Taylor y Fourier.    

---

## 📦 Instalación
> ⚠️ Asegurate de clonar e instalar todo el contenido en un solo lugar para evitar errores de importación.

### 🌐 Uso en Google Colab

1. Cloná e instalá directamente desde la notebook:
   ```python
    !git clone https://github.com/Boots-bots/Settings.git
    %cd Settings
    !pip install -e .

2. Ahora podés usarla
   ```python
    from settings.ajustes import Minimizer


### 🔧 Uso local (VSC o similar)

1. Cloná el repositorio:
   ```bash
   git clone https://github.com/Boots-bots/Settings.git
   cd Settings

2. (Opcional) Creá y activá un entorno virtual:
   ```bash
   python -m venv venv
   source venv/Scripts/activate (o bin)  

3. Instalá la librería en modo editable:
   ```bash
   pip install -e .

4. Ahora podés usarla desde cualquier proyecto local:
    ejemplo:
    from settings.funciones import maximos
    from settings.series import serie_taylor

---

## 🛠️ Requisitos

El archivo `setup.py` incluye las dependencias necesarias. Algunas comunes:

- `numpy`
- `matplotlib`
- `scipy`
- `sympy`

Si falta alguna al correr, podés instalarla manualmente:
   ```bash
    pip install nombre_de_la_libreria

---

## 🧪 Ejemplo de rápido uso
   ```python
    from settings.imports import *
    from settings.interpolacion import interpol_lagrange

    x = [0, 1, 2]
    y = [0, 1, 0]
    f = interpol_lagrange(x, y)
    ejex = np.linspace(0,2,20)
        
    fig, ax = plt.subplots()
    ax.plot(ejex, f(ejex), "b")
    ax.plot(x,y,"ro")
    ax.grid()
    plt.show()
