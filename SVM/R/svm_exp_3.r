library(RWeka)
library(e1071)
library(caret)

data<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")

data$Age<-as.numeric(as.factor(data$age))
data$Income<-as.numeric(as.factor(data$income))
data$Student<-as.numeric(as.factor(data$student))
data$CreditRating<-as.numeric(as.factor(data$credit_rating))

kfolds<-createFolds(dat$buys_computer,k=5)
sumKF=0
for(i in kfolds){
  train<-data[-i,]
  test<-data[i,]
  model<-svm(buys_computer~.,data=train)
  prediction<-predict(model,test)
  cfMatrix<-confusionMatrix(data=prediction,test$buys_computer)
  sumKF<-sumKF+cfMatrix$overall[1]
}
accuracy<-sumKF/length(kfolds)
accuracy

