# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:58:42 2018

@author: tchat
"""
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from plotSVMBoundaries import plotDecBoundaries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

wine_feature_train = r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\feature_train.csv'
wine_label_train = r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\label_train.csv'

X_train = np.genfromtxt(wine_feature_train, delimiter=",")[:,0:2]
y_train = np.genfromtxt(wine_label_train, delimiter=",")

skf = StratifiedKFold(n_splits = 5, shuffle = True)
Cs = np.logspace(-3, 3, 50)
gammas = np.logspace(-3, 3, 50)

ACC = np.zeros((50,50))
DEV = np.zeros((50,50))

for i, gamma in enumerate(gammas):
    for j, C in enumerate(Cs):  
        acc = []
        for train_index, dev_index in skf.split(X_train, y_train):
            X_cv_train, X_cv_dev = X_train[train_index], X_train[dev_index]
            y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
            clf = SVC(C = C, kernel = 'rbf', gamma = gamma, )
            clf.fit(X_cv_train, y_cv_train)
            acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))
        
        ACC[i,j] = np.mean(acc)
        DEV[i,j] = np.std(acc)
        
plt.imshow(ACC, interpolation = 'nearest', cmap=plt.cm.Blues)
plt.colorbar()

i, j = np.argwhere(ACC == np.max(ACC))[0]
print('The best pair is C = ' + str(Cs[j]) + ' and gamma = ' + str(gammas[i]))
print('The mean Cross-Validation Accuracy for the best pair = ', ACC[i,j])
print('The Standard deviation for the best pair = ', DEV[i,j])
print('')

