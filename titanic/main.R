setwd("../kaggle/titanic/")

library(rpart)
library(rattle)
library(tree)
library(caret)
library(randomForest)
library(magrittr)
library(stringr)
library(tidyr)
library(data.table)
library(dplyr)

ds_train <- read.csv("train.csv")
ds_test <- read.csv("test.csv")

ds_train_minus <- ds_train %>%
    select(-Survived)

ds <- rbind(ds_train_minus,ds_test)

ds <- ds %>%
    mutate(bkp_name = Name, Family_Size = SibSp + Parch + 1)

ds <- ds %>%
    separate(bkp_name,
             c("lastname", "leftover"),
             extra = "drop",
             fill = "right",
             sep = ", ") %>%
    separate(leftover,
             c("Title", "leftover"),
             extra = "drop",
             fill = "right",
             sep = " ") %>%
    select(-lastname, -leftover)

ds$Title <- str_replace(ds$Title,"[.]","")

ds$Title[ds$Title=="the"] <- "Other"
ds$Title[ds$Title=="Ms"] <- "Miss"
ds$Title[ds$Title=="Sir"] <- "Other"
ds$Title[ds$Title=="Jonkheer"] <- "Mr"
ds$Title[ds$Title=="Don"] <- "Other"
ds$Title[ds$Title=="Dona"] <- "Other"
ds$Title[ds$Title=="Capt"] <- "Other"
ds$Title[ds$Title=="Lady"] <- "Other"
ds$Title[ds$Title=="Major"] <- "Other"
ds$Title[ds$Title=="Col"] <- "Other"
ds$Title[ds$Title=="Dr"] <- "Other"
ds$Title[ds$Title=="Mlle"] <- "Other"
ds$Title[ds$Title=="Mme"] <- "Other"
ds$Title[ds$Title=="Rev"] <- "Other"

rows_na <- as.numeric(row.names(ds[is.na(ds$Fare),]))
ds$Fare[rows_na] <- median(ds$Fare, na.rm = T)

ds$Embarked[c(62,830)] <- "S"

predicted_age <- rpart(Age ~
                           Pclass +
                           Sex +
                           SibSp +
                           Parch +
                           Fare +
                           Embarked +
                           Title +
                           Family_Size,
                       data = ds[!is.na(ds$Age),],
                       method = "anova")

ds$Age[is.na(ds$Age)] <- predict(predicted_age, ds[is.na(ds$Age),])

ds$AgeGrp[ds$Age<7] <- "Children"
ds$AgeGrp[ds$Age>6 & ds$Age<18] <- "Teen"
ds$AgeGrp[ds$Age>17 & ds$Age<50] <- "Adult"
ds$AgeGrp[ds$Age>59] <- "Mature"

ds$Ticket <- substr(ds$Ticket,1,1)
ds$Cabin <- substr(ds$Cabin,1,1)

ds_train_temp <- ds[1:891,]
ds_test <- ds[892:1309,]

ds_train <- ds_train_temp %>%
    mutate(Survived = ds_train$Survived)

set.seed(8)

model_forest <- ds_train %>%
    randomForest(as.factor(Survived) ~
                     Fare +
                     Pclass +
                     Sex +
                     Age +
                     Family_Size,
                 data = .,
                 importance=T)

varImpPlot(model_forest)

model_tree <- ds_train %>%
    rpart(Survived ~
              Fare +
              AgeGrp +
              Pclass +
              Family_Size +
              Title +
              Cabin,
          data=.,
          method="class")

fancyRpartPlot(model_tree)

model_caret_rf <- ds_train %>%
    train(as.factor(Survived) ~
              Title +
              AgeGrp +
              Family_Size +
              Fare +
              Pclass +
              ,
          data = .,
          method = "rf",
          trControl = trainControl(method="repeatedcv",
                                   number=5,
                                   repeats=5))

model_caret_gbm <- ds_train %>%
    train(as.factor(Survived) ~
              Pclass +
              Family_Size  +
              Fare +
              Age +
              Sex +
              Embarked +
              Title,
          data = .,
          method = "gbm",
          trControl = trainControl(method="repeatedcv",
                                   number=10,
                                   repeats=5),
          verbose=F
          )

model_caret_svm <- ds_train %>%
    train(as.factor(Survived) ~
              Pclass +
              Family_Size  +
              Fare +
              Age +
              Sex +
              Embarked +
              Title,
          data = .,
          method = "svmRadial",
          trControl = trainControl(method="repeatedcv",
                                   number=10,
                                   repeats=5),
          tuneLength = 8)

model_caret_ctree <- ds_train %>%
    train(as.factor(Survived) ~
              Pclass +
              Family_Size  +
              Fare +
              AgeGrp +
              Embarked +
              Title +
              Cabin,
          data = .,
          method = "ctree",
          trControl = trainControl(method="repeatedcv",
                                   number=10,
                                   repeats=5),
          tuneLength = 8)


resample_models <- resamples(list(RF = model_caret_rf,
                                  GBM = model_caret_gbm,
                                  SVM = model_caret_svm,
                                  CTREE = model_caret_ctree))

difvalues = diff(resample_models)
summary(difvalues)

varImp(model_caret_ctree)
varImp(model_caret_gbm)
varImp(model_caret_rf)
varImp(model_caret_svm)

ds_test_predicted <- predict(model_tree, newdata = ds_test)
ds_test_predicted <- ifelse(ds_test_predicted=="0",0,1)
ds_test_predicted <- ifelse(ds_test_predicted[,2]<0.65,0,1)

ds_test_answer <- data.frame(PassengerId = c(892:1309),
                             Survived = ds_test_predicted)

write.csv(x = ds_test_answer, file = "ds_test_answer.csv", row.names = F)

