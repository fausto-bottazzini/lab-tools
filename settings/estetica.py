" codigo para el estilo de los graficos "

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (11,6)
plt.rcParams.update({'font.size': 14})

from matplotlib.pyplot import style                                        # graficos lindos
# plt.style.use('seaborn-v0_8')
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.ion() 


