library(RWeka)
library(e1071)
library(caret)

#read dataset & testdata 
train<-read.csv("/home/onu/Desktop/ML/clustering/Data/data.csv")
test<-read.csv("/home/onu/Desktop/ML/clustering/Data/testdata.csv")
#detach the class column put it in different array


x<-subset(train, select = -buys_computer)
y <- train[,"buys_computer"]

#svm <- SVM(x, y, core="libsvm", kernel="linear", C=1)

#bind the train data and test data into a table so that it will help to make dummy variable for these strings
data<-rbind(x,test)
#making dummy
data$age<-as.numeric(as.factor(data$age))
data$income<-as.numeric(as.factor(data$income))
data$student<-as.numeric(as.factor(data$student))
data$credit_rating<-as.numeric(as.factor(data$credit_rating))
#as there are one test data which is at thelast row so first 14 data detach from it and put in x variable
x<-data[1:14,]
#detach the test data
test<-data[15:15,]
#reset indexing of test data
rownames(test) <- NULL
#make dataframe of training data
dat = data.frame(x, buys_computer = as.factor(y))
# make model
model <-svm(buys_computer~., data=dat)
#predict the class
pred <-predict(model, test)
#print predict
pred


