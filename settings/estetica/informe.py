" codigo para el estilo de los graficos tipo informe (en blanco)"

import matplotlib.pyplot as plt
from matplotlib import rcParams

def estilo_informe():
    """Estilo neutro para informes científicos"""
    rcParams.update(plt.rcParamsDefault)  # Resetea a estilo por defecto

    # Opcional: ajustes menores
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 12
    rcParams["axes.labelsize"] = 12
    rcParams["axes.titlesize"] = 14