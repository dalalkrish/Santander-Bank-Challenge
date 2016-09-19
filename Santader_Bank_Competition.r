#R_script

setwd("/Users/Krishnang/Documents/Data Mining Case/krish")

ini_train <- read.csv('Train.csv', header = T)
ini_test <- read.csv('Test.csv', header = T)

ini_train1 <- ini_train[,-1]
ini_test1 <- ini_test[,-1]

sapply(test, function(x) sum(is.na(x)))
sapply(ini_test1, function(x) sum(is.na(x)))

a <- ncol(ini_train1)
b <- ncol(ini_test1)

#Replacing missing values with column means
for (i in 1:a) {
  
 ini_train1[is.na(ini_train1[,i]),i] <- mean(ini_train1[,i], na.rm = T) 
}

for (i in 1:b) {
  
  ini_test1[is.na(ini_test1[,i]),i] <- mean(ini_test1[,i], na.rm = T) 
}

x <- ini_train1[,-c(26:29)]
y <- ini_test1[,-c(26:29)]

#Visualizing the Data
for (i in 1:ncol(x)) {
  c <- unique(x[,i])
  d <- length(c)
  if (d > 20){
    
  hist(c, freq = T, main = paste("Total unique values", d, sep = ':'),
         xlab = paste("feaure", i, sep = ':'))
}
  else {
    barplot(table(unique(x[,i])),col = rainbow(20),
            main = paste("Total unique values", d, sep = ':'), xlab = paste("feature", i, sep = ':'))
  }}

#Scaling the Data
library(caret)
preProc_x <- preProcess(x[,-252], method = "knnImpute", thresh = 0.80, verbose = F)
x_1 <- predict(preProc_x, x)
y_1 <- predict(preProc_x, y)

train <- cbind.data.frame(ini_train1[,26:29], x_1)
test <- cbind.data.frame(ini_test1[26:29],y_1)

library(plyr)
train[,1] = mapvalues(train$Trans24, from = c("Enable", "Not-Enable"), to = c("1", "0"))
train[,2] = mapvalues(train$Trans25, from = c("Enable", "Not-Enable"), to = c("1", "0"))
train[,3] = mapvalues(train$Trans26, from = c("Enable", "Not-Enable"), to = c("1", "0"))
train[,4] = mapvalues(train$Trans27, from = c("Enable", "Not-Enable"), to = c("1", "0"))
train[,5] = mapvalues(train$Cust_status, from = c("Old", "New"), to = c("1", "0"))

test[,1] = mapvalues(test$Trans24, from = c("Enable", "Not-Enable"), to = c("1", "0"))
test[,2] = mapvalues(test$Trans25, from = c("Enable", "Not-Enable"), to = c("1", "0"))
test[,3] = mapvalues(test$Trans26, from = c("Enable", "Not-Enable"), to = c("1", "0"))
test[,4] = mapvalues(test$Trans27, from = c("Enable", "Not-Enable"), to = c("1", "0"))
test[,5] = mapvalues(test$Cust_status, from = c("Old", "New"), to = c("1", "0"))

# library(outliers)
# train[,-c(1:5,256)] <- rm.outlier(train[,-c(1:5,256)], fill = T, opposite = T)
# test <- rm.outlier(test[,-c(1:5)], fill = T, opposite = T)

#Applying ML algorithms
library(e1071)
model <- svm(Active_Customer~., data = train, na.action = na.omit)
pred <- predict(model, test)
pred
new_svm <- ifelse(pred >= 0.5, 1, 0)
table(new_svm)
out_df <- cbind.data.frame(ini_test$Cust_id,new_svm)

log.model <- glm(Active_Customer~., family = binomial(link = "logit"),
                 data = train)
log.pred <- predict(log.model, test, type = "response", level = 0.95)
new_logit <- ifelse(log.pred > 0.5, 1, 0)
table(new_logit)
out.log <- cbind.data.frame(ini_test$Cust_id, new_logit)

library(randomForest)
m <- randomForest(Active_Customer~., data = train, ntree=100, norm.votes = F)
m.pred <- predict(m, test)
new_rf <- ifelse(m.pred >= 0.5, 1,0)
table(new_rf)

#Using Ensemble method to achieve better accuracy
assemble <- cbind.data.frame(ini_test$Cust_id, new_svm, new_logit, new_rf)
assemble = within(assemble, {
  final.col = ifelse(assemble[,2]+assemble[,3]+assemble[,4] >= 2, 1,0)
})

out <- cbind.data.frame(ini_test$Cust_id, assemble$final.col)
write.csv(out, "ensemble.csv")
