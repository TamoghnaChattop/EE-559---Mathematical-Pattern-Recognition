import numpy as np
import matplotlib.pyplot as plt
from plotSVMBoundaries import plotDecBoundaries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import svm

wine_feature_train = r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\feature_train.csv'
wine_feature_test = r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\feature_test.csv'
wine_label_train = r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\label_train.csv'
wine_label_test = r'C:\Users\tchat\.spyder-py3\HW 10\wine_csv\label_test.csv'

X_train = np.genfromtxt(wine_feature_train, delimiter=",")[:,0:2]
X_test = np.genfromtxt(wine_feature_test, delimiter=",")[:,0:2]
y_train = np.genfromtxt(wine_label_train, delimiter=",")
y_test = np.genfromtxt(wine_label_test, delimiter=",")

skf = StratifiedKFold(n_splits = 5, shuffle = True)
Cs = np.logspace(-3, 3, 50)
gammas = np.logspace(-3, 3, 50)

accuracy = np.zeros((50,50))
dev = np.zeros((50,50))

pair_history = [0, 0]
ACC = []
DEV = []

for t in range(0, 20):
    for i, gamma in enumerate(gammas):
        for j, C in enumerate(Cs):
            acc = []
            for train_index, dev_index in skf.split(X_train, y_train):
                X_cv_train, X_cv_dev = X_train[train_index], X_train[dev_index]
                y_cv_train, y_cv_dev = y_train[train_index], y_train[dev_index]
                clf = SVC(C = C, kernel = 'rbf', gamma = gamma, decision_function_shape = 'ovr')
                clf.fit(X_cv_train, y_cv_train)
                acc.append(accuracy_score(y_cv_dev, clf.predict(X_cv_dev)))
            
            accuracy[i,j] = np.mean(acc)
            dev[i,j] = np.std(acc)   
    i, j = np.argwhere(accuracy == np.max(accuracy))[0]
    pair_history = np.vstack([pair_history, [gammas[i], Cs[j]]])
    ACC.append(accuracy[i,j])
    DEV.append(dev[i,j])

pair_history = pair_history[1:]
print('The 20 chosen pairs = \n', pair_history)
i = np.argwhere(ACC == np.max(ACC))[0]
print('The best pair is gamma = ' + str(pair_history[int(i),0]) + ' and C = ' + str(pair_history[int(i),1]))
print('The mean Cross-Validation Accuracy for the best pair = ', ACC[int(i)])
print('The Standard deviation for the best pair = ', DEV[int(i)])
print('')

# Part d
clf1 = SVC(C = pair_history[int(i),0], kernel = 'rbf', gamma = pair_history[int(i),1], decision_function_shape = 'ovr')
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Test accuracy  = ', acc)
plotDecBoundaries(X_train, y_train, clf1)