import pandas as pd
from sklearn.preprocessing import LabelEncoder

bankdata = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
X = bankdata.drop('buys_computer', axis=1)
y = bankdata['buys_computer']
bankdata1 = pd.read_csv("/home/onu/Desktop/ML/clustering/Data/testdata.csv")
Z = bankdata1

X=X.append(Z,ignore_index=True)

from sklearn import preprocessing, metrics

le = LabelEncoder()
X['age']=le.fit_transform(X['age'])
X['income']=le.fit_transform(X['income'])
X['student']=le.fit_transform(X['student'])
X['credit_rating']=le.fit_transform(X['credit_rating'])
Z=X.drop(X.index[:-1]).reset_index(drop=True)
X=X[:-1]

y=le.fit_transform(y)
#print(X)

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils.vis_utils import plot_model
X_train=X.to_numpy()
Y_train = y
X_test = Z.to_numpy()
# create model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='sigmoid'))

model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, epochs=55,batch_size=5,verbose=2)
scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

Y_pred = model.predict(X_test)
pred_res=''
for i in range(0,len(Y_pred)):
    if Y_pred[i]>=0.5:
        pred_res='yes'
    else:
        pred_res='no'
print(pred_res,Y_pred)
