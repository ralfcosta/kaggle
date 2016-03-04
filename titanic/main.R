setwd("../kaggle/titanic/")

library(rpart)
library(rattle)
library(tree)
library(caret)
library(magrittr)

ds_train <- read.csv("titanic/train.csv")
ds_test <- read.csv("titanic/test.csv")

decision_tree <- ds_train %>%
    rpart(Survived ~ Fare + Age + Sex + Pclass + SibSp + Parch, data=., method="class")

fancyRpartPlot(decision_tree)

ds_test_predicted <- predict(decision_tree, newdata = ds_test)
ds_test_predicted[,2] <- ifelse(ds_test_predicted[,2]<0.65,0,1)

ds_test_answer <- data.frame(PassengerId = c(892:1309),
                             Survived = ds_test_predicted[,2])

write.csv(x = ds_test_answer, file = "ds_test_answer.csv", row.names = F)
