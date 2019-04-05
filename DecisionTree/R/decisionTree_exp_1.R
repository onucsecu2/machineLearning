library(RWeka)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
train<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
test<-read.csv("/home/onu/Desktop/ML/clustering/Data/testdata.csv")
model<-J48(buys_computer~.,data=train)
prediction<-predict(model,test)
prediction


