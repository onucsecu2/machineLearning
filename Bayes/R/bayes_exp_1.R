library(RWeka)
library(caret)
library(e1071)
train<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
test<-read.csv("/home/onu/Desktop/ML/clustering/Data/testdata.csv")
model<-naiveBayes(buys_computer~.,data=train)
prediction<-predict(model,test)
prediction



