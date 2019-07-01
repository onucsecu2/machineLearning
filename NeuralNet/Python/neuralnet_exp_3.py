import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
bankdata = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
X = bankdata.drop('buys_computer', axis=1)
y = bankdata['buys_computer']

#X['age']= pd.get_dummies(X['age'])
le = LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(activation= 'logistic',early_stopping =True,max_iter=1000,learning_rate_init =0.1)

scores = []
cv = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index)
    print("Test Index: ", test_index, "\n")
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
sum=0 
print("accuracy : ")   
print(np.mean(scores))
