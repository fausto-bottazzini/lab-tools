" codigo para el estilo de los graficos "

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Tamaño del gráfico y fondo oscuro
rcParams["figure.figsize"] = (8, 8)
rcParams["figure.facecolor"] = "black"
rcParams["axes.facecolor"] = "black"
rcParams["savefig.facecolor"] = "black"

# Estilo de líneas y texto
rcParams["lines.linewidth"] = 2
rcParams["lines.color"] = "yellow"
rcParams["text.color"] = "white"
rcParams["axes.labelcolor"] = "white"
rcParams["xtick.color"] = "white"
rcParams["ytick.color"] = "white"
rcParams["axes.edgecolor"] = "white"

# Fuente y tamaño
rcParams["font.family"] = "serif"
rcParams["font.size"] = 16
rcParams["axes.labelsize"] = 14
rcParams["axes.titlesize"] = 18
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12

def configurar_estilo_axes(ax):
    """
    Aplica estilo Merlino a un objeto Axes:
    - Bordes izquierdo e inferior visibles
    - Bordes superior y derecho ocultos
    - Grilla punteada blanca y suave
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    ax.grid(True, linestyle=':', linewidth=0.5, color='white', alpha=0.3)
