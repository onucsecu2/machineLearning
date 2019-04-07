library(RWeka)
library(caret)
library(e1071)
library(nnet)
data<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
kfolds<-createFolds(data$buys_computer,k=5)
sumNN=0
for(i in kfolds){
  train<-data[-i,]
  test<-data[i,]
  model<-multinom(buys_computer~.,data=train)
  prediction<-predict(model,test)
  cfMatrix<-confusionMatrix(data=prediction,test$buys_computer)
  sumNN<-sumNN+cfMatrix$overall[1]
}
accuracy<-sumNN/length(kfolds)
accuracy




