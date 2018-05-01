#Melissa Laurino
#2/27/18
#Machine Learning - Linear Regression Project #3

#Write a linear regression algorithm from scratch in R, which prints coefficients and accuracy metrics 
##(e.g. R2) and plots the actual versus predicted response variables. Compare the R2 of the results using 
#your algorithm to the R2 of the results using lm() and predict(). 
#Document each step of the code to demonstrate you understand what each line of code does. 
#The code has to describe the steps of the linear regression model. 
#For your algorithm, you may use basic statistical functions and graphing functions but NOT machine 
#learning functions (i.e. no lm(), no predict() and related functions). 
#Note, you must fully explain what solve() is doing if you use it 
#(i.e. an explanation for a general audience more than
#  “The function solves the equation a %*% x = b for x, where b is a matrix”). 
#The deliverable is the well-documented code.

#Linear Regressions are used to find correlations, not necessarily causations.
###Is the data linear? Linear regressions figures out what the best line is to draw on the graph.
###Simple Linear Regressions - Only one x-variable
###Multiple Regressions - More than one x-varalbe
###B=Weights
###R^2 is a measure of association; a result closer to 1 means they are associated, but a result closer to 0 means not so much
###SSE=RSS; Sum of Squared Errors=Residual Sum of Squares; R^2: 1-(RSS/TSS)

#Linear Regression Steps:
##-Read in the data
##-Divide data into training and testing data
##-Train the model using the training dataset
###-Create X Matrix
###-Create Y Matrix
###-Solve for B Vector and Y^
##Make predictions using testing dataset
##Test model
##Compare R^2 of the results using your algorithm to the R2 of the results using lm() and predict(). 
##Plot predicted Y VS actual Y - Should be pretty linear

#Set working directory:
setwd("/Users/Melissa/Documents/GradSchool/MachineLearning/Regression") 

#Read in the data:
#Instead of loading library tidyverse and doing read_csv, use built in R
#Load in provided data and label it as a data frame, there areheaders
LaurinoData <- read.csv("TrainData_Laurino.csv", header = TRUE)

#Retrieve the dimension of the dataframe
dim(LaurinoData)
#1000obs of 6 variables; X1-X5 and Y

#Quick plot just to see what the data looks like:
#histogram(LaurinoData)
#with(LaurinoData, plot(X1, X2, xlab="X Data",ylab="Y Data",main="Laurino Train Data"))

#Divide the data into training and testing data:
#https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
#Training and Testing - Random Sample (using sample())
samples <- sample(1:nrow(LaurinoData), size=0.8*nrow(LaurinoData))
#Name the value sample so it does not overlap with sample()
training <- LaurinoData[samples,] #Take 800 rows or 80%
testing <- LaurinoData[-samples,] #Take the -rest of the rows (200)

dim(training)
#800 obs of 6 variables

dim(testing)
#200 obs of 6 variables

rm(samples) #Just to keep environment clean

#Separate X and Y variables of training and testing data for analysis
trainingX  <- subset(training , select = -c(Y) )
#Select all columns except the Y column - This will make up the X values for training.
trainingY  <- subset(training , select = c(Y) )
#Select only the Y column of training sample

testingX  <- subset(testing , select = -c(Y) )
#Select all columns except the Y column - This will make up the X values for testing.
testingY  <- subset(testing , select = c(Y) )
#Select only the Y column of testing sample

#Training Model using Training data - https://github.com/capt-calculator/linear-regression-from-scratch-r/blob/master/linear_regression_scratch.R
training_R <- function(x, y) { #A function of only x and y data
  intercept <- rep(1,nrow(x)) #To add an extra column of all 1's for the same amount of rows in x data
  x <- cbind(intercept, x) #Insert intercept column to X
  #From all rows i of training(x) data and for k variables
  matrix_X <- as.matrix(x) #Create X matrix from x data
  vector_Y <- as.matrix(y) #Create Y matrix/vector from y data
  beta <- solve(t(matrix_X) %*% matrix_X) %*% t(matrix_X) %*% vector_Y #The Y matrix is actually a vector (Link above & powerpoint), 
  #t=transpose as a matrix or take the inverse since they are of equal values, We want to multiply the original matrix by the transposed version
  #to retrieve the inverse - https://www.quora.com/What-is-the-difference-between-matrix-inverse-and-matrix-transpose
  #The beta is the result from the matrix multiplication - %*% must be used because we want the product of matrices, not standard multiplcation.
  beta <- round(beta, 2) #Round to 2 decimal places
  message(paste("Beta:", beta)) #Print/paste text in addition to the beta numerical values when you run the function in the console
  return(beta) #Solves for the β vector
} #Ends training_R

#Make predictions using Test data - https://github.com/capt-calculator/linear-regression-from-scratch-r/blob/master/linear_regression_scratch.R
#Similar to training model, but now we use the beta that we solved for in the training to get Y^
testing_R <- function(x,beta){
  beta_matrix <- t(as.matrix(beta)) #Transpose beta matrix
  intercept <- rep(1,nrow(x)) #To add an extra column of all 1's for the same amount of rows in x data
  x <- cbind(intercept, x) #Combine intercept with x as the new x
  matrix_X <- t(as.matrix(x)) #Transpose x data matrix
  y.hat <- beta_matrix %*% matrix_X #Ycarrot/hat - Multiply the beta matrix by the x data matrix
  #The data used is the training data to retrieve Y^
  message(paste("Y^:", y.hat))#Print/paste text in addition to the RSS numerical value when you run the function in the console
  return(y.hat) #Y^ is returned from the function
} #Ends testing_R

#Compute the beta using the training X and Y data using the function for training regression that we just created:
beta <- training_R(trainingX, trainingY)
dim(beta) #Results are 6 values of 
print(beta)
#            Y
#intercept  2.93
#X1        -0.04
#X2         2.21
#X3        -0.07
#X4         4.84
#X5         4.58

#Compute the Y^ with testing regression function using the X test data and the beta values retrieved from the training regression
y.hat <- testing_R(testingX, beta)
dim(y.hat) #Results are 200 values from the test regression
print(y.hat)
#Results printed in console

#Test the model: Compare our results with R's built in function, compute R^2
errors <- function(Y, y.hat_data){ #Even though we need to use y data and y.hat data, 
  #we can not use the same variables as before or the function will produce errors and use the wrong values.
  #Make sure the Y data is being read as the matrix(vector) like vector_Y
  Y <- as.matrix(Y)
  y.hat_data <- t(as.matrix(y.hat_data))
  #Understanding relationship of R^2/RSS/TSS: https://www.riskprep.com/component/exam/?view=exam&layout=detail&id=131
  RSS <- 0 #Residual Sum of Squares
  #res <- numeric(length = length(y)) #https://swcarpentry.github.io/r-novice-inflammation/15-supp-loops-in-depth/
  for (y in seq_along(Y)) { #?seq_along ^^^ - To be used for matrices
    RSS = RSS + (Y[y,1]-y.hat_data[y,1])**2 #Take the matrix y.hat values and subtract it from the Y data and square the result
    }#Error in Y[Y, 1] : only 0's may be mixed with negative subscripts
    #Error in Y[Y, 1] : incorrect number of dimensions
  message(paste("RSS:", RSS)) #Print/paste text in addition to the RSS numerical value when you run the function in the console
  y_hat_real <- mean(y.hat_data) #Retrieve the mean value of the Y^ generated values
  TSS <- 0 #The Total Sum of Squares
  #Y <- numeric(length = length(y)) #https://swcarpentry.github.io/r-novice-inflammation/15-supp-loops-in-depth/
  for (y in seq_along(Y)){ #Similar to above
    TSS = TSS + (Y[y,1]-y_hat_real)**2 #The actual value of y minus the mean, squared
  }
  message(paste("TSS:", TSS)) #Print/paste text in addition to the TSS numerical value when you run the function in the console
  r_squared <- 1 - (RSS/TSS) #Slide#58, 1 minus the RSS value divided by the TSS value
  message(paste("R^2:", r_squared)) #Print/paste text in addition to the R^2 numerical value when you run the function in the console
}#Ends errors function

errors(testingY,y.hat)
#Results printed in console:
#RSS: 1717.14380672746
#TSS: 40530.4970985205
#R^2: 0.957633290246762

#Before we create the function it was easier to understand the R results using the built in lm function with our data:
R_version_lm <- lm(formula = Y ~ X1 + X2 + X3 + X4 + X5, data = LaurinoData) #https://www.statmethods.net/stats/regression.html
summary(R_version_lm)
#The summary will produce residuals, various coefficients/errors at the intercepts, residual standard error, multiple
#R-squared, adjusted R-squared, F-statistic and a p-value.
?lm
?predict.lm #Function to produce predicted values used with regression

errorANDresults <- function(alldata){
  #The formula for lm is found in the powerpoint and also the resource on the next line. We have 5 X columns so we will use X1-X5, the
  #Y will depend~ on the X columns
  R_version_lm <- lm(formula = Y ~ X1 + X2 + X3 + X4 + X5, data = alldata) #https://www.statmethods.net/stats/regression.html
  #We know what these results look like because of summary(R_version_lm) used above, slide#20
  R_version_prediction <- predict.lm(R_version_lm, newdata=NULL, type="response") #slide21, use predict.lm instead of lm.pred, probabilities for the trianing data since newdata=NULL
  LaurinoData_Y <- LaurinoData[6] #All of the Y data from the original dataset (1,000obs)
  LaurinoData_Y <- as.matrix(LaurinoData_Y) #Read it in as a matrix to obtain lm R^2 
  y.hat_R <- as.matrix(R_version_prediction) #Y^ results from R's functions
  RSS <- 0 #Residual Sum of Squares
  for (y in seq_along(LaurinoData_Y)){ #Same as above
    RSS = RSS + (LaurinoData_Y[y,1]-y.hat_R[y,1])**2
  }
  message(paste("RSS:", RSS))
  y_hat_real <- mean(y.hat_R) #Take the mean of the Y^ R version matrix
  TSS <- 0 #Total Sum of Squares
  for (y in seq_along(LaurinoData_Y)){ #Same as above
    TSS = TSS + (LaurinoData_Y[y,1]-y_hat_real)**2 #The actual value of y minus the mean, squared
  }
  message(paste("TSS:", TSS))
  lm_r_squared <- 1 - (RSS/TSS) #R^2 is 1 minus the RSS divided by the TSS from above
  message(paste("lm R^2:", lm_r_squared))
}#Ends errorANDresults function

errorANDresults(LaurinoData)
#RSS 10232.9853670105
#TSS 241160.726061606
#lm R^2: 0.95756777841017

print(errors(testingY,y.hat), errorANDresults(LaurinoData))
#RSS: 1717.14380672746     RSS: 10232.9853670105
#TSS: 40530.4970985205     TSS: 241160.726061606
#R^2: 0.957633290246762 lm R^2: 0.95756777841017

#In conclusion, both R^2 values from the regression from scratch and also using he built in functions with R are close to 
#1 which means they values are associated, but are also very close to each other. 
#It is interesting to see how different the RSS/TSS values are but the R^2 value is still extremeley close...
#We now need to graph the association:

"The plotting refers to the actual Y values against the predicted Y values"
testingY <- as.matrix(testingY) #Actual Y values
PY <- testing_R(testingX, beta) #Name the results something different this time..
PY <- as.matrix(PY) #..because they need to be a matrix; Predicted Y values, #Does not work if they are not matrices
plot(x=testingY, y=PY, col=2, pch=1, font.lab=2, main="Actual Testing Y Values VS. Predicted Values",xlab = "Testing Y Values", ylab="Predicted Values")
#The results are linear with a few outliers. The outliers are to blame for the R^2 results not being even closer to 1.



#OLD/NOTES
lm(formula = Y ~ Size + Lot, data = LaurinoData)
RSS= RSS + (Y[y,1]-y.hat[y,1])**2
TSS=TSS + (Y[y,1]-y.hat)**2  
r_squared=1-(RSS/TSS)
R_version_lm <- lm(formula = Y ~ X1 + X2 + X3 + X4 + X5, data = LaurinoData) #https://www.statmethods.net/stats/regression.html
summary(R_version_lm)
#R's Function:
fit <- lm(Y ~ X1 + X2 + X3 + X4, + X5, data=LaurinoData)
summary(fit) # show results
plot(lm(testingY,PY))
