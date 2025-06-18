# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:05:11 2025

@author: jy129
"""

import sklearn.datasets as d
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import  sklearn.model_selection as ms
import pandas as pd

import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate


X, y = d.make_moons(n_samples=3000, noise=0.16, random_state=432) #달 형태

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=100)

#선형 분리: clf = classifier 분류가 
svm_clf = svm.SVC(kernel='linear', random_state=100)



scores = cross_val_score(svm_clf, X, y, cv=5) # cross variable folds

print("score====", scores)

print(pd.DataFrame(cross_validate(svm_clf, X, y,cv=5)))
             
print("score====", scores)

print("mean score===", scores.mean() )



#비선형 분리: clf = classifier 분류가 
svm_clf = svm.SVC(kernel='rbf', random_state=100)



scores = cross_val_score(svm_clf, X, y, cv=5) # cross variable folds

print("score====", scores)

print(pd.DataFrame(cross_validate(svm_clf, X, y,cv=5)))
             
print("score====", scores)

print("mean score===", scores.mean() )



from sklearn.model_selection import GridSearchCV #corss validation

svm_clf = svm.SVC(kernel = 'rbf', random_state=100) # 난수지정
parm = {'C': [0.001, 0.01, 0.1, 1, 10,25,50,100],
        'gamma': [ 0.001, 0.01, 0.1, 1,10,25,50,100]}

grid_svm = GridSearchCV(svm_clf, param_grid=parm, cv=5)

grid_svm.fit(X_train, y_train)

result = pd.DataFrame(grid_svm.cv_results_['params'])
result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False)

