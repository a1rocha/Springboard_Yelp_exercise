# Project with adjustments after presentation

# Packages needed for this project

# install.packages("FNN")
# install.packages("RCurl")
# install.packages("MASS")
# install.packages("leaps")
# install.packages("PerformanceAnalytics")
# install.packages("class")
# install.packages("gmodels")
# install.packages("psych")
# install.packages("party")
# install.packages("caret")
# install.packages("e1071")
library(FNN)
library(RCurl)
library(MASS)
library(leaps)
library(PerformanceAnalytics)
library(class)
library(gmodels)
library(psych)
library(party)
library(caret)
library(e1071)

# Multi Linear Regression

# renaming/making copy of the original dataset and removing the date attribute 
# because we have all the specific characteristics of the each day, so that variable 
# becomes disposable
bike10at <- SeoulBikeData[2:11]

# checking the amount of NAs in the dataset
numberNAs <- sum((is.na(SeoulBikeData)))
# 0 NAs in the dataset

# checking the summary of the dataset
summary(SeoulBikeData)

# checking the structure of the dataset
str(SeoulBikeData)

# visualize chart.correlation() with 10 attributes
chart.Correlation(bike10at, histogram = TRUE, method = "pearson")

# testing different methods as parameter in the function above first "kendall" and then "spearman"
chart.Correlation(bike10at, histogram = TRUE, method = "kendall")
chart.Correlation(bike10at, histogram = TRUE, method = "spearman")

# first we need to normalize the data to minimize errors due to different scales of measured variables
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

bikepartial_n <- as.data.frame(lapply(bike10at, normalize))

# using set.seed to have reproducible results
set.seed(1000)

# splitting dataset into train (70%) and test (30%)
# we do not want the train and test sets to have any data points in common
b_train <- sample(nrow(bikepartial_n), floor(nrow(bikepartial_n)*0.7))
train10 <- bikepartial_n[b_train,]
test10 <- bikepartial_n[-b_train,]

# creating linear model using 10 attributes
model_mlr10 <- lm(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
                    Visibility+DewPointTemperature+SolarRadiation+
                    Rainfall+Snowfall, data = bikepartial_n)
summary(model_mlr10)
anova(model_mlr10)

# prediction using 10 attributes
prediction10 <- predict(model_mlr10, interval = "prediction", newdata = test10)

# calculating the errors for 10 attributes
errors10 <- prediction10[,"fit"] - test10$RentedBikeCount
hist(errors10)
plot(density(errors10))

# calculating rmse - root mean squared errors for 10 attributes
rmse10 <- sqrt(sum((prediction10[,"fit"]-test10$RentedBikeCount)^2)/nrow(test10))
rmse10

# defining parameters for stepwise regression
full10 <- lm(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
               Visibility+DewPointTemperature+SolarRadiation+
               Rainfall+Snowfall, data = bikepartial_n)
null10 <- lm(RentedBikeCount~1, data = bikepartial_n)

# applying forward selection
stepf10 <- stepAIC(null10, scope = list(lower=null10, upper=full10), direction = "forward", trace = TRUE)
summary(stepf10)

# applying backward elimination
stepb10 <- stepAIC(full10, direction = "backward", trace = TRUE)
summary(stepb10)

# applying stepwise regression
stepw10 <- stepAIC(null10, scope = list(lower=null10, upper=full10), direction = "both", trace = TRUE)
summary(stepw10)

# checking the best combination between 10 attributes
subsets10 <- regsubsets(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
                          Visibility+DewPointTemperature+SolarRadiation+
                          Rainfall+Snowfall, data = bikepartial_n, nbest = 1)
sub.sum10 <- summary(subsets10)

all10 <- regsubsets(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
                      Visibility+DewPointTemperature+SolarRadiation+
                      Rainfall+Snowfall, data = bikepartial_n, nbest = 1, nvmax = 10)
info10 <- summary(all10)
table10 <- cbind(info10$which, round(cbind(rsq=info10$rsq, adjr2=info10$adjr2, cp=info10$cp, bic=info10$bic, 
                                rss=info10$rss), 3))
table10

#knn regression partial model

# making a copy of the data using only numeric attributes excluding the dependent variable
knnReg_partial <- bike10at

# making the outcome its own object
RentedBikeOutcomePartialM <- knnReg_partial$RentedBikeCount

# using normalize function in the dataset
knnReg_partial_n <- as.data.frame(lapply(knnReg_partial[2:10], normalize))

#binding the target variable to the dataset
knnReg_partial_n <- cbind(RentedBikeOutcomePartialM, knnReg_partial_n)

# using set.seed to have reproducible results
set.seed(1000)

# splitting dataset into train (70%) and test (30%)
# we do not want the train and test sets to have any data points in common
# dividing the dataset into test and train sets
indexKNNp <- sample(nrow(knnReg_partial_n), 0.70*nrow(knnReg_partial_n))
trainKNNpart <- knnReg_partial_n[indexKNNp,]
testKNNpart <- knnReg_partial_n[-indexKNNp,]

# creating the labels
trainKNNpart_labels <- trainKNNpart[,1]
testKNNpart_labels <- testKNNpart[,1]

# building the regression function
regKNNp <- knn.reg(trainKNNpart[,2:10], testKNNpart[,2:10], trainKNNpart[,1], k = 78)
print(regKNNp)

# plotting the predicted results printed above against the actual values of the outcome variable test set
plot(testKNNpart[,1], regKNNp$pred, xlab = "y", ylab = expression(hat(y)))

# mean squared errors prediction for knn partial model
MSEknnpartial <- mean((testKNNpart_labels - regKNNp$pred) ^ 2)
RMSEknnpartial <- sqrt(MSEknnpartial)

#mean absolute errors prediction for knn partial model
MAEknnpartial <- mean(abs(testKNNpart_labels - regKNNp$pred))

# slicing dataframe to work with 13 attributes
bike13at <- SeoulBikeData[2:14]

# the attributes listed as factor will need to be dummy coded in order for the regression to work
# dummy code variables that have two levels are coded 1/0
bike13at$Holiday <- ifelse(bike13at$Holiday == "Yes", 1, 0)
bike13at$FunctioningDay <- ifelse(bike13at$FunctioningDay == "Yes", 1, 0)

# dummy code variables that have three or more levels
Seasons_dummy13MLR <- as.data.frame(dummy.code(bike13at$Seasons))

# combining the dummy variables to the dataset
bike13at <- cbind(bike13at, Seasons_dummy13MLR)

# removing the original variable with more three or more levels
bike13at <- bike13at[-11]

# visualize chart.correlation() with 13 attributes
chart.Correlation(bike13at, histogram = TRUE, method = "pearson")

# testing different methods as parameter in the function above first "kendall" and then "spearman"
chart.Correlation(bike13at, histogram = TRUE, method = "kendall")
chart.Correlation(bike13at, histogram = TRUE, method = "spearman")

# first we need to normalize the data to minimize errors due to different scales of measured variables
bikefull_n <- as.data.frame(lapply(bike13at, normalize))

# using set.seed to have reproducible results
set.seed(1000)

# splitting dataset into train (70%) and test (30%)
# we do not want the train and test sets to have any data points in common
c_train <- sample(nrow(bikefull_n), floor(nrow(bikefull_n)*0.7))
train13 <- bikefull_n[c_train,]
test13 <- bikefull_n[-c_train,]

# creating linear model using 13 attributes
model_mlr13 <- lm(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
                    Visibility+DewPointTemperature+SolarRadiation+
                    Rainfall+Snowfall+Holiday+FunctioningDay+
                    Spring+Summer+Autumn+Winter, data = train13)
summary(model_mlr13)
anova(model_mlr13)

# prediction using 13 attributes
prediction13 <- predict(model_mlr13, interval = "prediction", newdata = test13)
# Warning message:
# In predict.lm(model_mlr13, interval = "prediction", newdata = test13) :
# prediction from a rank-deficient fit may be misleading

# calculating the errors for 10 attributes
errors13 <- prediction13[,"fit"] - test13$RentedBikeCount
hist(errors13)
plot(density(errors13))

# calculating rmse - root mean squared errors for 13 attributes
rmse13 <- sqrt(sum((prediction13[,"fit"]-test13$RentedBikeCount)^2)/nrow(test13))
rmse13

# defining parameters for stepwise regression
full13 <- lm(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
               Visibility+DewPointTemperature+SolarRadiation+
               Rainfall+Snowfall+Holiday+FunctioningDay+Spring+
               Summer+Autumn+Winter, data = bikefull_n)
null13 <- lm(RentedBikeCount~1, data = bikefull_n)

# applying forward selection
stepf13 <- stepAIC(null13, scope = list(lower=null13, upper=full13), direction = "forward", trace = TRUE)
summary(stepf13)

# applying backward elimination
stepb13 <- stepAIC(full13, direction = "backward", trace = TRUE)
summary(stepb13)

# applying stepwise regression
stepw13 <- stepAIC(null13, scope = list(lower=null13, upper=full13), direction = "both", trace = TRUE)
summary(stepw13)

# checking the best combination between 13 attributes
subsets13 <- regsubsets(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
                          Visibility+DewPointTemperature+SolarRadiation+
                          Rainfall+Snowfall+Holiday+FunctioningDay+
                          Spring+Summer+Autumn+Winter, data = bikefull_n, nbest = 1)
sub.sum13 <- summary(subsets13)

all13 <- regsubsets(RentedBikeCount~Hour+Temperature+Humidity+WindSpeed+
                      Visibility+DewPointTemperature+SolarRadiation+
                      Rainfall+Snowfall+Holiday+FunctioningDay+
                      Spring+Summer+Autumn+Winter, data = bikefull_n, nbest = 1, nvmax = 16)
info13 <- summary(all13)
table13 <- cbind(info13$which, round(cbind(rsq=info13$rsq, adjr2=info13$adjr2, cp=info13$cp, bic=info13$bic, 
                                           rss=info13$rss), 3))
table13

# knn regression full model

# making a copy of the data using only numeric attributes excluding the date
knnReg_full <- SeoulBikeData
knnReg_full <- knnReg_full[-1]

# making the outcome its own object
RentedBikeOutcomeFullM <- knnReg_full$RentedBikeCount

# the attributes listed as factor will need to be dummy coded in order for the regression to work
# dummy code variables that have two levels are coded 1/0
knnReg_full$Holiday <- ifelse(knnReg_full$Holiday == "Yes", 1, 0)
knnReg_full$FunctioningDay <- ifelse(knnReg_full$FunctioningDay == "Yes", 1, 0)

# dummy code variables that have three or more levels
Seasons_dummyKNN_MLR <- as.data.frame(dummy.code(knnReg_full$Seasons))

# combining the dummy variables to the dataset
knnReg_full <- cbind(knnReg_full, Seasons_dummyKNN_MLR)

# removing the original variable with more three or more levels
knnReg_full <- knnReg_full[-11]

# using normalize function in the dataset
knnReg_full_n <- as.data.frame(lapply(knnReg_full[2:16], normalize))

#binding the target variable to the dataset
knnReg_full_n <- cbind(RentedBikeOutcomeFullM, knnReg_full_n)

# using set.seed to have reproducible results
set.seed(1000)

# splitting dataset into train (70%) and test (30%)
# we do not want the train and test sets to have any data points in common
# dividing the dataset into test and train sets
indexKNNf <- sample(nrow(knnReg_full_n), 0.70*nrow(knnReg_full_n))
trainKNNfull <- knnReg_full_n[indexKNNf,]
testKNNfull <- knnReg_full_n[-indexKNNf,]

# creating the labels
trainKNNfull_labels <- trainKNNfull[,1]
testKNNfull_labels <- testKNNfull[,1]

# building the regression function # k was chosen based on being roughly the square root of the sample size
# in this case 6132 for the train set
regKNNf <- knn.reg(trainKNNfull[,2:16], testKNNfull[,2:16], trainKNNfull[,1], k = 78)
print(regKNNf)

# plotting the predicted results printed above against the actual values of the outcome variable test set
plot(testKNNfull[,1], regKNNf$pred, xlab = "y", ylab = expression(hat(y)))

# mean squared errors prediction for knn partial model
MSEknnfull <- mean((testKNNfull_labels - regKNNf$pred) ^ 2)
RMSEknnfull <- sqrt(MSEknnfull)

#mean absolute errors prediction for knn partial model
MAEknnfull <- mean(abs(testKNNfull_labels - regKNNf$pred))

#*****************************************************************************************#

# Machine Learning

# excluding date for dataset, dividing the RentedBikeCount attribute
# into 15 factors (intervals of 240 each) for a better prediction
# the dataset ranges roughly from 0 to 3600 and creating the RentedBikeFactor attribute to be
# used as the variable to be predicted
# working will all the attributes but Seasons, Holiday and FunctioningDay
bike11atML <- SeoulBikeData[2:11]

# make a copy of RentedBikeCount and binding it to the dataset as RentedBikeFactor
RentedBikeFactor <- SeoulBikeData$RentedBikeCount
bike11atML <- cbind(RentedBikeFactor, bike11atML)

# using cut function to transform the RentedBikeCount into a factor
bike11atML$RentedBikeFactor <- cut(SeoulBikeData$RentedBikeCount, breaks = seq(0,3600, by = 240),
                                   labels = c("1","2","3","4","5","6","7","8","9","10","11","12",
                                              "13","14","15"),
                                   include.lowest = TRUE)


# creating the normalize function
# with normalization , we can utilize our database for further queries and analysis
# we increase our data consistency by fixing our ranges between 0 -1 for all variables
# the function will be applied to all the attributes but the "RentedBikeFactor"
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

# binding and renaming column to RentedBikeFactor
bike11at_n <- as.data.frame(lapply(bike11atML[2:11], normalize))
bike11at_n <- cbind(bike11atML$RentedBikeFactor, bike11at_n)
colnames(bike11at_n)[colnames(bike11at_n) == 'bike11atML$RentedBikeFactor'] <- 'RentedBikeFactor'


# dividing the dataset into test and train sets
set.seed(555)
index11 <- sample(nrow(bike11at_n), 0.70*nrow(bike11at_n))
train11knn <- bike11at_n[index11,]
test11knn <- bike11at_n[-index11,]

# creating labels and applying KNN technique

train11knn_labels <- train11knn[,1]
test11knn_labels <- test11knn[,1]

# we have to decide on the number of neighbors (k). There are several rules of thumb, 
# one being the square root of the number of observations in the training set. 
# In this case, we select 78 or 79 as the number of neighbors, which is approximately the square 
# root of our sample size N = 6132
train11knn_pred <- knn(train = train11knn[,2:11], test = test11knn[,2:11], cl = train11knn[,1], k = 78)

# constructing the confusion matrix: the order is TN || FP || FN || TP

cm11 <- table(Actual = test11knn_labels, Predicted = train11knn_pred)
cm11

# this function divides the correct predictions by total number of predictions that tell 
# us how accurate the model is.

accuracy_func<- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy_func(cm11)

#picking the optimal number of neighbors (k) for the model with 10 attributes
RBF10_pred_caret <- train(train11knn[,2:11], train11knn[,1], method = "knn",
                          preProcess = c("center", "scale"))
RBF10_pred_caret

# k = 9 for optimal model, that is when kappa peaked
knnPredict11 <- predict(RBF10_pred_caret, newdata = test11knn[,2:11])

ConfM11 <- confusionMatrix(knnPredict11, test11knn_labels)
ConfM11

# ML working with all the attributes in the dataset
bike14atML <- SeoulBikeData[2:14]

# the attributes listed as factor will need to be dummy coded in order for the regression to work
# dummy code variables that have two levels are coded 1/0
bike14atML$Holiday <- ifelse(bike14atML$Holiday == "Yes", 1, 0)
bike14atML$FunctioningDay <- ifelse(bike14atML$FunctioningDay == "Yes", 1, 0)

# dummy code variables that have three or more levels
Seasons_dummy14knn <- as.data.frame(dummy.code(bike14atML$Seasons))

# combining the dummy variables to the dataset
bike14atML <- cbind(bike14atML, Seasons_dummy14knn)

# removing the original variable with more three or more levels
bike14atML <- bike14atML[-11]

# make a copy of RentedBikeCount, naming it RentedBikeFactor and binding it to the dataset
RentedBikeFactor <- SeoulBikeData$RentedBikeCount
bike14atML <- cbind(RentedBikeFactor, bike14atML)

# using cut function to transform the RentedBikeFactor into a factor
# it was numeric previously due to it being a copy of RentedBikeCount
bike14atML$RentedBikeFactor <- cut(SeoulBikeData$RentedBikeCount, breaks = seq(0,3600, by = 240),
                                   labels = c("1","2","3","4","5","6","7","8","9","10","11","12",
                                              "13","14","15"),
                                   include.lowest = TRUE)

# utilizing the normalize function once again
# binding and renaming column to RentedBikeFactor
bike14at_n <- as.data.frame(lapply(bike14atML[2:17], normalize))
bike14at_n <- cbind(bike14atML$RentedBikeFactor, bike14at_n)
colnames(bike14at_n)[colnames(bike14at_n) == 'bike14atML$RentedBikeFactor'] <- 'RentedBikeFactor'

# dividing the dataset into test and train sets
set.seed(777)
index14 <- sample(nrow(bike14at_n), 0.70*nrow(bike14at_n))
train14knn <- bike14at_n[index14,]
test14knn <- bike14at_n[-index14,]

# creating labels and applying KNN technique

train14knn_labels <- train14knn[,1]
test14knn_labels <- test14knn[,1]

# we have to decide on the number of neighbors (k). There are several rules of thumb, 
# one being the square root of the number of observations in the training set. 
# In this case, we select 78 as the number of neighbors, which is approximately the square 
# root of our sample size N = 6132
train14knn_pred <- knn(train = train14knn[,2:17], test = test14knn[,2:17], cl = train14knn[,1], k = 78)

cm14 <- table(Actual = test14knn_labels, Predicted = train14knn_pred)
cm14

#utilizing the accuracy_func once again
accuracy14 <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy14(cm14)


#picking the optimal number of neighbors (k) for the full model
RBF14_pred_caret <- train(train14knn[,2:17], train14knn[,1], method = "knn",
                          preProcess = c("center", "scale"))
RBF14_pred_caret

# k = 5 for optimal model, that is when kappa peaked

knnPredict14 <- predict(RBF14_pred_caret, newdata = test14knn[,2:17])

ConfM14 <-confusionMatrix(knnPredict14, test14knn_labels)
ConfM14
#********************************************************************************#
