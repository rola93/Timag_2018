import numpy as np
import matplotlib.pyplot as plt
import random

seed=123
np.random.seed(seed)
random.seed(seed)

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df



# Build a classification task using 3 informative features
X = np.load('vectors/X_train.npy')
y = np.load('vectors/y_train.npy')

print("X.shape=",X.shape)
X =np.delete(X, np.array([2,3,4,7,8,9,12]),1)
print("X.shape=",X.shape)
#['Red', 'Green','Blue','Gmod','H','S','V','L','A', 'B', 'Op0', 'Op1','Op2']
#sacamos [Blue,Gmod, H,L,A,B, 02]

"""
X_train, y_train = make_classification(n_samples=100,
                           n_features=10,
                           n_informative=4,
                           n_redundant=1,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
"""

from sklearn.metrics import make_scorer, f1_score

print(f"De los {len(X)} ejemplos disponibles utilizamos {len(X)//2}")

X = X[len(X)//2:]
y = y[len(y)//2:]

param_grid = dict(
    clf__n_estimators=[50,100],
    clf__min_samples_split=[200],
    clf__max_depth=[5,10]
)
pipe = Pipeline([
    ('clf', RandomForestClassifier(criterion='gini',random_state=123, n_jobs=1))
])

gs = GridSearchCV(pipe, param_grid=param_grid, scoring=make_scorer(f1_score, labels=[0,1], pos_label=1), n_jobs=-1, cv=2, verbose=10, refit=False)

print("Comienza el entrenamiento")
gs.fit(X,y)

del(X)
del(y)

print("Best parameters according to grid search:")
for k in gs.best_params_:
    print("\t{}: {}".format(k, gs.best_params_[k]))
print("----------------")

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
    
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        param_string = '{' + ', '.join(['{}: {}'.format(k, params[k]) for k in sorted(params)]) + '}'

        print("\t%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, param_string))
print("Final de la ejecicion")
