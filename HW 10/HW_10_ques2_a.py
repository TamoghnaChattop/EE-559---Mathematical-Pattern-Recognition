# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:55:14 2018

@author: tchat
"""

from sklearn.svm import SVC
from sklearn import svm
import csv
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from plotSVMBoundaries import plotDecBoundaries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import math

df1 = pd.read_csv(r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\feature_train.csv',header=None)
x=df1.as_matrix()
x=x[:,0:2:1]
df2 = pd.read_csv(r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\label_train.csv',header=None)
y=df2.as_matrix()
y=np.ravel(y)
x,y=shuffle(x,y)
all_acc=[]
save_avg_acc=[]
skf=StratifiedKFold(n_splits=5,shuffle=True)
cnt=1
for train_index, test_index in skf.split(x,y):
    x_train, x_test=x[train_index], x[test_index]
    y_train, y_test=y[train_index], y[test_index]
    clf=SVC(C=1.0, kernel='rbf', gamma=1.0)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    all_acc.append(acc)
    cnt+=1
mean=np.mean(all_acc)
print('average cross validation accuracy: ', mean)
