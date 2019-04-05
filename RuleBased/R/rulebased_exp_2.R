library(RWeka)
library(caret)

data<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
train<-data[1:10,1:5]
test<-data[11:14,1:5]
model<-OneR(buys_computer~.,data=train)
prediction<-predict(model,test)
prediction
cfMatrix<-confusionMatrix(data=prediction,test$buys_computer)
cfMatrix
