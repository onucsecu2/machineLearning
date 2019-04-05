
import pandas as pd
from sklearn.preprocessing import LabelEncoder

bankdata = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
X = bankdata.drop('buys_computer', axis=1)
y = bankdata['buys_computer']



from sklearn import preprocessing, metrics

#X['age']= pd.get_dummies(X['age'])
le = preprocessing.LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


