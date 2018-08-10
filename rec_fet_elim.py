import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.feature_selection import RFE

seed=123
np.random.seed(seed)
random.seed(seed)

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# Build a classification task using 3 informative features
"""
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=1,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
"""
X = np.load('vectors/X_train.npy')
y = np.load('vectors/y_train.npy')

print("X.shape=",X.shape)
X =np.delete(X, np.array([7,8,9]),1)
print("X.shape=",X.shape)

forest = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=10,
                              min_samples_split=20,
                              random_state=0, n_jobs=-1)

rfe = RFE(estimator=forest, n_features_to_select=1, step=1, verbose=10)
rfe.fit(X, y)
ranking = rfe.ranking_

print(ranking)

importances =  X.shape[1] / ranking

indices = np.argsort(importances)[::-1]

columns = np.array(['Red', 'Green','Blue','Gmod','H','S','V', 'Op0', 'Op1','Op2'])
#columns = np.array(['Red', 'Green','Blue','Gmod','H','S','V','L','A', 'B', 'Op0', 'Op1','Op2', 'label'])
print(columns)

# Print the feature ranking
print("Feature ranking:")

for idx, f in enumerate(columns[indices]):
    print("%d. feature %s " % (idx + 1, f))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), columns[indices])
plt.xlim([-1, X.shape[1]])
plt.savefig(r"RFE_RF.png")
