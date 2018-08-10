import numpy as np
import random

seed=123
np.random.seed(seed)
random.seed(seed)

from pandas import DataFrame as df

import matplotlib.pyplot as plt

print("leyendo datos...")
X_train = np.load('vectors/X_train.npy')
y_train = np.load('vectors/y_train.npy')

print("X_train.shape=",X_train.shape)
X_train = np.delete(X_train, np.array([7,8,9]),1)
print("X.shape=",X_train.shape)


columns = np.array(['Red', 'Green','Blue','Gmod','H','S','V', 'Op0', 'Op1','Op2'])

assert len(columns) == X_train.shape[1]

distances=np.empty((len(columns),), dtype=np.float32)

for idx, axis_x in enumerate(columns):

    p, _, _ = plt.hist(X_train[y_train==0,idx], alpha=0.5, color='r', label='Tejido',  cumulative=False, density=True)
    q, _, _ =     plt.hist(X_train[y_train==1,idx], alpha=0.5, color='g', label='Instrumento',  cumulative=False, density=True)

    plt.title("Histograma {}".format(axis_x))
    plt.savefig(f"figs/dens_{axis_x}.png")

    distances[idx] = -np.log(np.sqrt(p*q).sum())
    plt.clf()


print("Feature ranking:")

indices = np.argsort(distances)[::-1]

for idx, f in enumerate(columns[indices]):
    print("%d. feature %s (%f)" % (idx + 1, f, distances[indices[idx]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), distances[indices],
       color="r", align="center")
plt.xticks(range(X_train.shape[1]), columns[indices])
plt.xlim([-1, X_train.shape[1]])
plt.savefig(r"b_distance.png")
