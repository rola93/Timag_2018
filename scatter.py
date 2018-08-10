import numpy as np
import random

seed=123
np.random.seed(seed)
random.seed(seed)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots

print("leyendo datos...")
X_train = np.load('vectors/X_train.npy')
y_train = np.load('vectors/y_train.npy')


columns = ['Red', 'Green','Blue','Gmod','H','S','V','L','A', 'B', 'Op0', 'Op1','Op2']

for idx, axis_x in enumerate(columns):
    for jdx, axis_y in enumerate(columns):
        print(f"Calculando para {axis_x}-{axis_y}")

        if idx == jdx:
            plt.hist(X_train[y_train==0,idx], alpha=0.5, color='r', label='Tejido')
            plt.hist(X_train[y_train==1,idx], alpha=0.5, color='g', label='Instrumento')

            plt.title("Histograma {}".format(axis_x))
            plt.savefig(f"figs/{axis_x}-{axis_y}.png")
            plt.clf()
        elif idx < jdx:
            o, = plt.plot(X_train[y_train==0,idx], X_train[y_train==0,jdx], 'rx', alpha=0.5)
            i, = plt.plot(X_train[y_train==1,idx], X_train[y_train==1,jdx], 'gx', alpha=0.5)

            plt.legend([i, o], ["Instrumento", "Tejido"])

            plt.axis('equal')
            plt.title(f"Puntos en {axis_x}-{axis_y} para Instrumento - Tejido")
            plt.xlabel(axis_x)
            plt.ylabel(axis_y)
            plt.savefig(f"figs/{axis_x}-{axis_y}.png")
            plt.clf()

exit()
