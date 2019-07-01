import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
bankdata = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/data.csv")

X = bankdata.drop('buys_computer', axis=1)
y = bankdata['buys_computer']
from sklearn import preprocessing, metrics

le = LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
scores = []
cv = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index)
    print("Test Index: ", test_index, "\n")
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index], y.iloc[test_index]
    svclassifier.fit(X_train, y_train)
    scores.append(svclassifier.score(X_test, y_test))
sum=0 
print("accuracy : ")   
print(np.mean(scores))
