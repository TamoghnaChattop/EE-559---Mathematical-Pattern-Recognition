"""
@author: Pranav Gundewar
EE 559 Assignment 10
Q2 - Cross Validation Model for Support Vector Machines 
Data sets: HW10_1
"""

# Importing Libraries
import numpy as np 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Importing Training Data
X = np.genfromtxt("F:\Books\EE 559\HW 10\wine_csv\\feature_train.csv",delimiter=',') #Load Training data
X = X[:,:2]
y = np.genfromtxt("F:\Books\EE 559\HW 10\wine_csv\\label_train.csv",delimiter=',') #Load Training Labels

skf = StratifiedKFold(n_splits=5, shuffle=True)   #Stratified K fold for 5 folds preserving percentage of each class 
skf.get_n_splits(X, y)
'''
accuracy = 0
for train_index, test_index in skf.split(X, y): #Cross Vaidating Data used for Training SVM classifier for 5 folds
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   clf = SVC(gamma=1,kernel='rbf')  #create an object of SVM classifier
   clf.fit(X_train, y_train)   #Train the classifier
   acc_score = clf.score(X_train, y_train) #Calculate Training Accuracy
   acc_score *= 100
   accuracy +=acc_score

print('Average Cross-Validation Accuracy: ',accuracy/skf.get_n_splits(X, y))     #Calculating Mean accuracy over n folds
'''
acc = np.empty([50,50])
std = np.empty([50,50])
g = np.logspace(-3,3)
c = np.logspace(-3,3)
for i in range (len(g)):
    for j in range (len(c)):
         a = np.zeros(5)
         clf = SVC(gamma=g[i],C=c[j])  #create an object of SVM classifier
         for train_index, test_index in skf.split(X, y): #Cross Vaidating Data used for Training SVM classifier for 5 folds
             k = 0
             X_train, X_test = X[train_index], X[test_index]
             y_train, y_test = y[train_index], y[test_index]             
             clf.fit(X_train, y_train)   #Train the classifier
             a[k]=clf.score(X_train, y_train)
             k +=1
         acc[i,j] = np.mean(a)
         std[i,j] = np.std(a)
         
(P,Q) = np.meshgrid(g, c)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(P, Q, acc, cmap='viridis', edgecolor='none')

             
             
             
             
             
             
             
             