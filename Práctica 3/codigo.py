# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:37:01 2018

@author: malve
"""
from __future__ import print_function

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn import svm

#Cargamos los conjuntos de datos
X_train = np.load("Datos/optdigits_tra_X.npy")
y_train =np.load("Datos/optdigits_tra_y.npy")
X_test=np.load("Datos/optdigits_tes_X.npy")
y_test=np.load("Datos/optdigits_tes_y.npy")


#Estandarizamos  los datos utilizando pipeline
pipe = Pipeline([('Scale',preprocessing.StandardScaler()),('Norm', preprocessing.Normalizer())])
pipe.fit(X_train)
X_train=pipe.transform(X_train)
X_test=pipe.transform(X_test)

#Creamos el clasificador 
parameters = {'kernel':('linear','rbf'), 'C':[1, 10, 100, 1000]}
svc = svm.SVC()

#Aprendemos a partir del conjunto de entrenamiento
clf = GridSearchCV(svc, parameters).fit(X_train,y_train)

#Predecimos los valores del conjunto de prueba
print()
print()
print()
print("Classification report for classifier %s:\n%s\n" 
      % (clf, metrics.classification_report(y_test, y_test_predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_test_predict))
print()
