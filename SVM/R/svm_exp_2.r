library(RWeka)
library(e1071)
library(caret)

data<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
#making dummy
data$age<-as.numeric(as.factor(data$age))
data$income<-as.numeric(as.factor(data$income))
data$student<-as.numeric(as.factor(data$student))
data$credit_rating<-as.numeric(as.factor(data$credit_rating))
#detaching training and test data
train<-data[1:10,1:5]
test<-data[11:14,1:5]
x<-subset(train, select = -buys_computer)
y <- train[,"buys_computer"]
x
y
#making dataframe
dat = data.frame(x, buys_computer = as.factor(y))
#making model
model <-svm(buys_computer~., data=dat)
#making prediction
pred <-predict(model, test)
#making confusion matrix
cfMatrix<-confusionMatrix(data=pred,test$buys_computer)
cfMatrix
