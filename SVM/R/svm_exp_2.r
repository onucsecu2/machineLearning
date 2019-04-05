library(RWeka)
library(e1071)
library(caret)

data<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
#making dummy
data$Age<-as.numeric(as.factor(data$age))
data$Income<-as.numeric(as.factor(data$income))
data$Student<-as.numeric(as.factor(data$student))
data$CreditRating<-as.numeric(as.factor(data$credit_rating))
#detaching training and test data
train<-data[1:10,1:5]
test<-data[11:14,1:5]

x<-subset(train, select = -buys_computer)
y <- train[,"buys_computer"]
#making dataframe
dat = data.frame(x, buys_computer = as.factor(y))
#making model
model <-svm(buys_computer~., data=dat)
#making prediction
pred <-predict(model, test)
pred
#making confusion matrix
cfMatrix<-confusionMatrix(data=pred,test$buys_computer)
cfMatrix
