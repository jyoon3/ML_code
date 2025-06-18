# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:45:13 2025

@author: jy129
"""

import sklearn.datasets as d
import pandas as pd

# dataset load
x = d.load_breast_cancer()

cancer = pd.DataFrame( data= x.data, columns=x.feature_names)

#print("cancer=", cancer)

cancer['target'] = x.target

#cancer.info()

#cancer.describe()
#cancer.target.value_counts()


# SVM 학습 

import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate

#선형 분리: clf = classifier 분류가 
svm_clf = svm.SVC(kernel='linear')

X = x.data
y = x.target


scores = cross_val_score(svm_clf, X, y, cv=5) # cross variable folds


print(pd.DataFrame(cross_validate(svm_clf, X, y,cv=5)))
             
print("score====", scores)

print("mean score===", scores.mean() )

svm_clf = svm.SVC(kernel='rbf')


scores = cross_val_score(svm_clf, X, y, cv=5) # cross variable folds

print(pd.DataFrame(cross_validate(svm_clf, X, y,cv=5)))
print("score====", scores)

print("Mean Score===", scores.mean() )

X = cancer.iloc[:, :-1]
y = cancer.iloc[:, -1 ]

from sklearn.preprocessing import StandardScaler
import  sklearn.model_selection as ms

# 스케일링
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = ms.train_test_split(X_scaled, y, test_size=0.3, random_state=100)


#선형 분리: clf = classifier 분류가 
svm_clf = svm.SVC(kernel='linear', random_state=100)

X = x.data
y = x.target


scores = cross_val_score(svm_clf, X_scaled, y, cv=5) # cross variable folds


print(pd.DataFrame(cross_validate(svm_clf, X_scaled, y,cv=5)))
             
print("score====", scores)

print("mean score===", scores.mean() )

from sklearn.model_selection import GridSearchCV #corss validation

svm_clf = svm.SVC(kernel = 'linear', random_state=100) # 난수지정
parm = {'C': [0.001, 0.01, 0.1, 1, 10,25,50,100]}

grid_svm = GridSearchCV(svm_clf, param_grid=parm, cv=5)

grid_svm.fit(X_train, y_train)

result = pd.DataFrame(grid_svm.cv_results_['params'])
result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False)

