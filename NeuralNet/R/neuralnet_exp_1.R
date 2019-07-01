library(RWeka)
library(caret)
library(e1071)
library(nnet)
library(neuralnet)
train<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
test<-read.csv("/home/onu/Desktop/ML/clustering/Data/testdata.csv")
model<-multinom(buys_computer~.,data=train)
prediction<-predict(model,test)
prediction
'''
head(train)
data<-train
data$age<-as.numeric(as.factor(data$age))
data$income<-as.numeric(as.factor(data$income))
data$student<-as.numeric(as.factor(data$student))
data$credit_rating<-as.numeric(as.factor(data$credit_rating))

head(data)
NN = neuralnet(buys_computer~ age+income+student+credit_rating, data, hidden = 3 , linear.output = T )
plot(NN)
'''

