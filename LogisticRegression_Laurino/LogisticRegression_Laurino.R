#Melissa Laurino
#Machine Learning - Logistic Regression Project 4
#4/17/17

#Write a logistic regression algorithm in R using the logit (or sigmoid) function from scratch: 
#-Print coefficients and accuracy metrics. 
#-Document each step of the code to demonstrate you understand what each line of code does. 
#-The code has to describe the steps of the logistic regression model. 
#-Then compare your algorithm to the algorithms in R using glm() and predict().

#Notes:
#Logistic regression is binomial distribution
#Regression - problems with a quantitative response
#Classification - problems with a qualitative response (linearly separable-decision boundaries)
#Sensitivity - measures the proportion of positives that are correctly identified (true positive rate)
#Specificity - Measures the proportion of negatives that are correctly identified (true negative rate)
#Accuracy - measures the proportion of the correct identifications

#Summary:
#-Read in the data
#-Divide data into training and testing data columns
#-Determine iterations, L
#-Train the model with TrainData
#--Create X matrix
#--Create Y matrix 
#--Create W matric of k zeros
#--Iterations loop
#-Make predictions using TestData
#-Test the model

#install.packages("caTools") #Library to split the dataset
library(caTools) #Library used for splitting datasets that you can assign a ratio to..but we already have the data split

#Set working directory:
setwd("/Users/Melissa/Documents/GradSchool/MachineLearning/LogisticRegression_Laurino") 

#Instead of loading library tidyverse and doing read_csv, use built in R
#Load in provided data and label it as a data frame, there are headers
TrainData <- read.csv("TrainData.csv", header = TRUE)
#Specify column headers or it will read them as data points.

dim(TrainData)
#1000 observations of 3 variables
#Viewing the dimensions of data are important because we are working with matrices

#Load in test data and label it as a data frame, there are headers
TestData <-read.csv("TestData.csv", header = TRUE)

dim(TestData)
#250 observations of 3 variables

#Separate vectors of testing and training data for logistic regression algorithm from scratch:
TrainDataX <- subset(TrainData , select = -c(Y) )
dim(TrainDataX)
#1000,2
TrainDataX_matrix <- t(as.matrix(TrainDataX)) #If we label the X data as a matrix before we run the function,
#the data is not able to add intercepts for the rows of the data.
dim(TrainDataX_matrix)
#2,1000 - We needed to use t() to reverse the data
TrainDataY <- TrainData$Y
TrainDataY_matrix <- t(as.matrix(TrainDataY))
dim(TrainDataY_matrix)
#1,1000

#Refer to the dim() or global environment to see the header names of the data rows for separating.
TestDataX <- subset(TestData , select = -c(Y) )
TestDataX_matrix <- t(as.matrix(TestDataX))
dim(TestDataX_matrix)
#2,250
TestDataY <- TestData$Y
dim(TestDataY)

#Determine number of iterations:
L <- 1000

#Develop sigmoid (logit) function from scratch: https://rpubs.com/junworks/Understanding-Logistic-Regression-from-Scratch & Slide 9
sigmoid <- function(z){
      g <- 1/(1+exp(z))
      return(g)
}
#h(x)=1/1+e(−x⋅θ)

#Logistic regression from scratch: https://rpubs.com/junworks/Understanding-Logistic-Regression-from-Scratch
logistic_regression <- function(x, y, L, alpha){ #Logistic regression needs 4 inputs unlike linear regression which was 2
  y <- as.matrix(y) #Ensure y is a matrix
  intercept <- rep(1, length(y)) #To add an extra column of all 1's for the same amount of rows repeated for the length of y data
  x <- cbind(intercept, x) #Insert intercept column to x data matrix
  x <- t(as.matrix(x)) #If we use the matrix X data, it does not know how to add the 1 intercepts. We have to use the regular
  #x data in the function and then specify it as a matrix after.
  w <- rep(0,nrow(x)) #Repeat zeros for every row in the length of x data to be our w vector
  w <- t(as.matrix(w)) #The t() is needed or error: non-conformable arguments
  m <- nrow(x) #m is the number of rows in the x data
  iterations <- 0 #Start the iterations at 0, but specify to not go over L below:
  while (iterations < L){ #While the iterations are less than 1,000 or whatever we specify L to be:
    #assign g as the result of the sigmoid function
    #the function is passed the dot-product of X and W
    g <- t(sigmoid(w%*%x)) #*Error non-comformable arguments, sigmoid is w%*%x, not the opposite - Use the sigmoid function on the matrix multiplication of w vector and x data
    #the gradient is computed, with a formula derived from the slides
    gradient <- t((x %*% (g - y))/m) #Error non-comformable arguments
    w <- (w - (alpha * gradient))
    iterations <- iterations + 1 #Start the counter for L
  }
  return(list(w=w, gradient=gradient)) #Return w and gradient
}

#Test the logistic regression function to find beta(b) with our data
alpha=0.5
b<-logistic_regression(TrainDataX,TrainDataY,L,alpha)
b
#message(paste("Beta:", b))
#$w
#intercept        X1       X2
#-60099.92    -1297.389 113828.7
#$gradient
#intercepts     X1        X2
#120.3333    2.660143 -227.7401

coefficients <- b$w
print(coefficients)
#intercepts        X1        X2
#-60099.92    -1297.389   113828.7

testing_logistic_regression <- function(x,y,w){ #Similar to linear regression:
  y <- as.matrix(y)#Y data needs to be read in as a matrix
  intercept <- rep(1, length(y)) #To add an extra column of all 1's for the length of y data
  #x <- rep(1,nrow(x)) #To add an extra column of all 1's for the same amount of rows in x data
  x <- cbind(intercept, x) #Combine intercept with x as the new x
  x <- t(as.matrix(x)) #Transpose x data matrix, we can not read in the matrix data or it will not allow us to add the intercepts
  YHAT <- t(1/(1 + exp(-1 * (w%*%x))))
  #y.hat <- sigmoid(w%*%x) #t(1/(1 + exp(-1 * (w%*%x)))) #Ycarrot/hat
  #The data used is the training data to retrieve Y^
  message(paste("Y^:", YHAT))#Print/paste text in addition to the RSS numerical value when you run the function in the console
  return(list(y.hat=YHAT, y=y)) #Y^ is returned from the function
} #Ends testing_logistic_regression
#It is very important that the return values do not match each other, (what you use in your data=what you call it in the function)

#This time, use the testing data to test logitistic regression for results, the important component here is Y^
prediction <- testing_logistic_regression(TestDataX, TestDataY, b$w) #Within b are the w and gradient values, but we only want the w values
prediction

?glm #Fitting Generalized Linear Models
#glm is used to fit generalized linear models, specified by giving a symbolic description of the linear predictor 
#and a description of the error distribution.
#glm(formula, family = gaussian, data, weights, subset,na.action, start = NULL, etastart, mustart, offset,control = list(...), model = TRUE, method = "glm.fit", x = FALSE, y = TRUE, contrasts = NULL, ...)

model1 <- glm(TrainDataY~., data=TrainData, family='binomial')
summary(model1) #Error, algorithm did not converge? Summary still below:
#Deviance Residuals: 
#  Min          1Q      Median          3Q         Max  
#-2.409e-06  -2.409e-06  -2.409e-06   2.409e-06   2.409e-06  

#Coefficients:
#  Estimate Std. Error z value Pr(>|z|)
#(Intercept) -2.657e+01  1.583e+04  -0.002    0.999
#X1           9.395e-15  1.105e+04   0.000    1.000
#X2          -8.185e-14  1.538e+04   0.000    1.000
#Y            5.313e+01  3.299e+04   0.002    0.999

#(Dispersion parameter for binomial family taken to be 1)

#Null deviance: 1.2794e+03  on 999  degrees of freedom
#Residual deviance: 5.8016e-09  on 996  degrees of freedom
#AIC: 8

#Number of Fisher Scoring iterations: 25

#R Prediction
?predict() #predict is a generic function for predictions from the results of various model fitting functions. 
#The function invokes particular methods which depend on the class of the first argument.
Rprediction <- predict(model1, newdata=TestData, type='response')
Rprediction
Rprediction_matrix <- as.matrix(Rprediction)

#Confusion Matrix - https://www.rdocumentation.org/packages/caret/versions/3.45/topics/confusionMatrix
library(ggplot2)
#Take a look at the original data:
print(ggplot(TrainData, aes(x = X1, y = X2, color = factor(TrainDataY))) + geom_point())
library(caret)
dim(Rprediction_matrix)
#250,1
dim(TestDataY)
TestDataY_matrix <- as.matrix(TestDataY)
dim(TestDataY_matrix)
#250,1
confusionMatrix(Rprediction_matrix, TestDataY) #Error: `data` and `reference` should be factors with the same levels.
#However, 250,1 are the dimensions for both objects...
#Let's try a different way to create a confusion matrix:
table(factor(Rprediction, levels=min(TestDataY):max(TestDataY)), factor(TestDataY, levels=min(TestDataY):max(TestDataY)))
#  0 1
#0 0 0
#1 0 0

classify <- function(data){ #From Louis Discenza..recomended to try classifying the data to have confusion matrix work..
  for (i in seq(nrow(data))){
    if (data[i] < 0.5)
      data[i] = 0
    else
      data[i] = 1
  }
  #return results
  return(data)
}

classify_Rprediction_matrix <- classify(as.matrix(Rprediction))
classify_Rprediction_matrix #Now all of the data points are 0's and 1's.
classify_TestDataY_matrix <- classify(as.matrix(TestDataY))
dim(classify_TestDataY_matrix)
caret::confusionMatrix(classify_Rprediction_matrix, classify_TestDataY_matrix)
#Error: `data` and `reference` should be factors with the same levels......
caret::confusionMatrix(classify_Rprediction_matrix, Rprediction$Y)
# Error: $ operator is invalid for atomic vectors 

prediction <- classify(prediction$y.hat)
prediction #Already cleaned/classified
TestDataY <- classify(TestDataY) #Already cleaned/classified
confusionMatrix(prediction, prediction$Y) #Error: `data` and `reference` should be factors with the same levels.
#Error in prediction$Y : $ operator is invalid for atomic vectors

#Since these attempts are still not working I am assumming this might be an R/package issue.

#Accuracy mesurements...also from scratch:
AccuracyMetrics <- function(y.hat, y){ #This will look at the number of correct and incorrect predictions made by our algorithm
  TP <- 0 #True Positive; #Initiate everything to start at 0 because we are starting iterations
  FP <- 0 #False Positive
  TN <- 0 #True Negatives
  FN <- 0 #False Negative
  Sensitivity <- 0 #True Positive Rate
  Specificity <- 0 #True Negative Rate
  Accuracy <- 0 #Correct Prediction Rate
  for (i in seq(nrow(y))){ #For each element in the sequence of y rows
    if (y.hat[i] == 1 & y[i] == 1){ #If y.hat is equal to 1 and the y value is equal to 1..
      TP <- TP + 1 #..then add one to the true positives
    }
    if (y.hat[i] == 1 & y[i] == 0){ #If y.hat is equal to 1 and the y value is equal to 0..
      FP <- FP + 1 #..then add one to the false positives
    }
    if (y.hat[i] == 0 & y[i] == 0){ #If y.hat is equal to zero and y is equal to zero..
      TN <- TN + 1 #..then add one to the true negatives
    }
    if (y.hat[i] == 0 & y[i] == 1){ #If y.hat is equal to 0 and y is equal to 1..
      FN <- FN + 1 #..then add one count to the false negatives
    }
  }
#Sensitivity/Specificity/Accuracy from Powerpoint (slide#23)
#Sensitivity
Sensitivity <- TP / (TP + FN) #True positives divided by (true positives + false negatives)
#Specificity
Specificity <- TN / (TN + FP) #True negatives divided by (true negatives + false positives)
#Accuracy
Accuracy <- (TP + TN) / (TP + TN + FN + FP) #True postives plus true negatives / true postives plus true negatives plus false negatives plus false positives

return(c(Sensitivity, Specificity, Accuracy))
}

prediction <- testing_logistic_regression(TestDataX, TestDataY, b$w)
prediction
classified_prediction <- classify(prediction$y)
classified_prediction
Accuracy_Metrics <- AccuracyMetrics(classified_prediction, prediction$y.hat)
Accuracy_Metrics
classified_prediction <- classify(prediction$y.hat)
classified_prediction
Accuracy_Metrics <- AccuracyMetrics(classified_prediction, prediction$y.hat)
Accuracy_Metrics
#[1] 0.7261905 0.9036145 0.8440000
#         1        1        1

#Sensitivity
print(Accuracy_Metrics[1])
#0.7261905
#1

#Specificity
print(Accuracy_Metrics[2])
#0.9036145
#1

#Accuracy
print(Accuracy_Metrics[3])
#0.8440000
#1
