import numpy as np
import matplotlib.pyplot as plt
import random

seed=123
np.random.seed(seed)
random.seed(seed)

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

# Build a classification task using 3 informative features
X = np.load('vectors/X_train.npy')
y = np.load('vectors/y_train.npy')

print("Borramos las features LAB porque son identicas a HSV")
print("X.shape=",X.shape)
X =np.delete(X, np.array([7,8,9]),1)
print("X.shape=",X.shape)

"""
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=4,
                           n_redundant=1,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
"""
from sklearn.model_selection import train_test_split

print(f"De los {len(X)} ejemplos disponibles utilizamos {len(X)//3}")
X_train, X_test, y_train, y_test =  train_test_split(X[:len(X)//3], y[:len(X)//3], test_size=0.8, random_state=123, stratify=y[:len(X)//3])
del(X)
del(y)
print(f"Entrenamos con {len(X_train)} y evaluamos con {len(X_test)}")

columns = np.array(['Red', 'Green','Blue','Gmod','H','S','V', 'Op0', 'Op1','Op2'])
n = len(columns)

final_x_labels = np.empty_like(columns)
score_by_step = np.empty_like(columns, dtype=np.float32)

for i in range(n):

    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10,
	                          min_samples_split=100,
	                          random_state=0, n_jobs=-1,
	                          verbose=0)

    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    score = f1_score(y_test, y_pred)
    importances = forest.feature_importances_
    del(forest)
    indices = np.argsort(importances)[::-1]
    idx_worst = indices[-1]

    # Print the feature ranking
    print(f"Feature ranking after removing {i} features:")

    for idx, f in enumerate(columns[indices]):
        print("%d. feature %s (%f)" % (idx + 1, f, importances[indices[idx]]))
    
    print(f"f1_score = {score}")

    final_x_labels[i] = columns[idx_worst]
    score_by_step[i] = score

    print(f"Removing feature {columns[idx_worst]}")
    X_train =np.delete(X_train, idx_worst,1)
    X_test =np.delete(X_test, idx_worst,1)
    columns =np.delete(columns, idx_worst,0)
    print(f"Now there are {X_train.shape[1]} features")



# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(n), score_by_step,
       color="r", align="center")
plt.xticks(range(n), final_x_labels)
plt.xlim([-1, n])
plt.savefig(r"Tracked_RFE_RF.png")

plt.clf()


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(n), score_by_step * 100,
       color="r", align="center")
plt.xticks(range(n), final_x_labels)
plt.xlim([-1, n])
plt.savefig(r"Tracked_RFE_zoom_RF.png")
