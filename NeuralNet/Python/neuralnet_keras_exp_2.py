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

'''
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
'''
from keras.models import Sequential
from keras.layers import Dense,Dropout
import keras
bankdata1=X.to_numpy()


X_train = bankdata1[0:10,0:4]
Y_train = bankdata1[0:10,4]
X_test = bankdata1[10:14,0:4]
Y_test = bankdata1[10:14,4]
print(X_train)
# create model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='sigmoid'))
model.add(Dropout(0.6))
model.add(Dense(4, activation='sigmoid'))
model.add(Dropout(0.6))
model.add(Dense(2, activation='sigmoid'))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, epochs=500,batch_size=32)
scores = model.evaluate(X_train, Y_train)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(X_test)
for i in range(0,len(Y_pred)):
    if Y_pred[i]>=0.5:
        Y_pred[i]=1
    else:
        Y_pred[i]=0
print(Y_pred)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

