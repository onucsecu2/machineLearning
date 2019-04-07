import pandas as pd
from sklearn.preprocessing import LabelEncoder

bankdata = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
X = bankdata.drop('buys_computer', axis=1)
y = bankdata['buys_computer']
bankdata1 = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/testdata.csv")
Z = bankdata1

X=X.append(Z,ignore_index=True)

from sklearn import preprocessing, metrics

le = preprocessing.LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])
Z=X.drop(X.index[:-1]).reset_index(drop=True)
X=X[:-1]

from sklearn.neural_network import

clf = MLPClassifier()
clf.fit(X, y)
y_pred = clf.predict(Z)
print(y_pred)


