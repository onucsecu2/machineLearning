library(RWeka)
library(caret)
library(e1071)
data<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
kfolds<-createFolds(data$buys_computer,k=5)
sumNB=0
for(i in kfolds){
  train<-data[-i,]
  test<-data[i,]
  model<-OneR(buys_computer~.,data=train)
  prediction<-predict(model,test)
  cfMatrix<-confusionMatrix(data=prediction,test$buys_computer)
  sumNB<-sumNB+cfMatrix$overall[1]
}
accuracy<-sumNB/length(kfolds)
accuracy




