import numpy as np
import matplotlib.pyplot as plt
import random

seed=123
np.random.seed(seed)
random.seed(seed)

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import time

# Build a classification task using 3 informative features
X = np.load('vectors/X_train.npy')
y = np.load('vectors/y_train.npy')

print("Borramos las features LAB porque son identicas a HSV")
print("X.shape=",X.shape)
X =np.delete(X, np.array([7,8,9]),1)
print("X.shape=",X.shape)

forest = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=10,
                              min_samples_split=100,
                              random_state=0, n_jobs=-1,
                              verbose=10)

print(f"Comienza el entrenamiento, usamos solo {len(X)//5} ejemplos de {len(X)} disponibles")
start = time.time()
forest.fit(X[:len(X)//5], y[:len(X)//5])
end = time.time()
print("El entrenamiento tomo {} secs".format(end - start))

n, fn = X.shape

del(X)
del(y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

del(forest)

indices = np.argsort(importances)[::-1]

columns = np.array(['Red', 'Green','Blue','Gmod','H','S','V', 'Op0', 'Op1','Op2'])
# columns = np.array(['Red', 'Green','Blue','Gmod','H','S','V','L','A', 'B', 'Op0', 'Op1','Op2'])

# Print the feature ranking
print("Feature ranking:")

for idx, f in enumerate(columns[indices]):
    print("%d. feature %s (%f)" % (idx + 1, f, importances[indices[idx]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(fn), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(fn), columns[indices])
plt.xlim([-1, fn])
plt.savefig(r"reduced_feature_importanceRF.png")
