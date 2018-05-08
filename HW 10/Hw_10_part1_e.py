# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:52:15 2018

@author: tchat
"""

import sklearn
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from plotSVMBoundaries import plotDecBoundaries

#Import train files and the label files
with open(r'C:\Users\tchat\.spyder-py3\HW 10\HW10_2_csv\train_x.csv') as feature_train:
    feature_training = csv.reader(feature_train)
    feature_list=[]
    for row in feature_training:
        if len (row) !=0:
            feature_list = feature_list + [row]

feat_train_data = np.array(feature_list).astype("float")

with open(r'C:\Users\tchat\.spyder-py3\HW 10\HW10_2_csv\train_y.csv') as label_train:
    label_training = csv.reader(label_train)
    label_list=[]
    for row in label_training:
        if len (row) !=0:
            label_list = label_list + [row]
            
label_train = []

for row in label_list:
    label_train.append(row[0])
label_train = np.array(label_train).astype("float")

#Train the Model
model = SVC(C=1.0, kernel='rbf', gamma=10)
model.fit(feat_train_data, label_train)
s = model.predict(feat_train_data)

acc= accuracy_score(label_train, s) 

plotDecBoundaries(feat_train_data, label_train, model)
