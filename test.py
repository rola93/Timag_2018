import numpy as np
import matplotlib.pyplot as plt
import random

seed=123
np.random.seed(seed)
random.seed(seed)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
import time

print("Comenzamos...\nLeemos los vectores")
X_train = np.load('vectors/X_train.npy').astype(np.int16)
y_train = np.load('vectors/y_train.npy').astype(np.uint8)

X_train = X_train[:len(X_train)//2]
y_train = y_train[:len(y_train)//2]

print("Eliminamos las features inutiles")
print("Antes: X_train.shape=",X_train.shape)
X_train = np.delete(X_train, np.array([2,3,4,7,8,9,12]),1)
print("Ahora: X_train.shape=",X_train.shape)
#['Red', 'Green','Blue','Gmod','H','S','V','L','A', 'B', 'Op0', 'Op1','Op2']
#sacamos [Blue, H,L,A,B, 02]

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

# ponemos los valores que encontramos en el grid search
clf = RandomForestClassifier(criterion='gini', random_state=123, n_jobs=-1, 
                             max_depth=10, min_samples_split=200, n_estimators=100, verbose=10)

print("Comienza el entrenamiento")

start = time.time()


clf.fit(X_train,y_train)
end = time.time()
print("El entrenamiento tomo {} secs".format(end - start))

del(X_train)
del(y_train)

print("Cargamos valores para test")
X_test = np.load('vectors/X_test.npy').astype(np.int16)

# seleccionamos las features que habiamos determinado
X_test = np.delete(X_test, np.array([2,3,4,7,8,9,12]),1)

print("Comienza el test")
start = time.time()
y_pred = clf.predict(X_test)
end = time.time()
print("El test tomo {} secs".format(end - start))
del(X_test)
print("generamos el reporte")
y_test = np.load('vectors/y_test.npy').astype(np.uint8)
print(classification_report(y_test, y_pred, target_names=['Tissue','instrument']))
del(y_pred)
print("Volvemos a cargar el test")
X_test = np.load('vectors/X_test.npy').astype(np.int16)
X_test = np.delete(X_test, np.array([2,3,4,7,8,9,12]),1)

p = clf.predict_proba(X_test)

y_score = p[:,1]

del(X_test)
del(p)



auc_roc = roc_auc_score(y_test, y_score)
print(f"auc_roc = {auc_roc}")

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
del(fpr)
del(tpr)
plt.legend(loc = 'best')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(f"ROC_test.png")
plt.clf()

precision, recall, _ = precision_recall_curve(y_test, y_score)
average_precision = average_precision_score(y_test, y_score)

del(y_score)
del(y_test)

print(f"auc_PR = {average_precision}")

plt.title('Precision-Recall')
plt.plot(recall, precision, 'b', label = 'AUC = %0.2f' % average_precision)
del(precision)
del(recall)
plt.legend(loc = 'best')
plt.plot([0, 1], [1, 0],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.savefig(f"PR_test.png")
plt.clf()

print("Guardando el modelo entrenado")

from sklearn.externals import joblib
from os.path import isdir
from os import listdir, makedirs
import re

if not isdir('./models'):
    version=0
else:
    version = max([int(s) for s in listdir('./models') if re.search(r'^\d+$',s)] + [0])

version +=1

model_dir = f'./models/{version}'

print(f"creando directorio para el modelo {model_dir}")

makedirs(model_dir)
print("Guardando el modelo")
joblib.dump(clf, model_dir+'/model.pkl')
print("Modelo guardado. Fin")





