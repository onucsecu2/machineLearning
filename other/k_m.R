library(RWeka)
library(caret)
library(e1071)
data<-read.csv("/home/onu/Desktop/ML/R lab/k_data.csv")
kmeans.result<-kmeans(data,2)
plot(data[c("X1","X2")],col=kmeans.result$cluster)
points(kmeans.result$centers[,c("X1","X2")],col=1:3,pch=8,cex=2)
SimpleKMeans(data,control = NULL)

