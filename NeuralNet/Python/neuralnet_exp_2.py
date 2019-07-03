import pandas as pd
from sklearn.preprocessing import LabelEncoder

bankdata = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
X = bankdata
#X = bankdata.drop('buys_computer', axis=1)
#y = bankdata['buys_computer']


from sklearn import preprocessing, metrics


le = LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])
X['buys_computer']=le.fit_transform(X['buys_computer'])

X_test=X.drop(X.index[:-4]).reset_index(drop=True)
y_test=y.drop(y.index[:-4]).reset_index(drop=True)
X_train=X[:-4]
y_train=y[:-4]

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

