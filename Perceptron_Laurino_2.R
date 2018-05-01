#Melissa Laurino
#Machine Learning - Perceptron Project 2
#2/19/17

#Write a perceptron algorithm from scratch, 
#Produce a graphical depiction of the data and accuracy metrics. 
#Describe the steps of the perceptron. 

#*Include accuracy measurements
#*Include graphs or visualizations of the "separated" data
#*The deliverable will be very well-documented R code and the output

#Notes::
#Perceptron is one of the most basice forms of a neural network.
#--It will have only two possible results; true/false, pos/neg, 0/1, etc. - Training data
#--Graphs are linearly seperable if a line divides the data into two sets, perceptron will find the line.
#---https://appliedgo.net/perceptron/
#Perceptron steps:
#--Create training data from original dataset
#--Create test data from original dataset
#--Run the perceptron function on the training data
#--Obtain accuracy of the weights from training data
#--Run the error function on the test data

#Set working directory:
setwd("/Users/Melissa/Documents/GradSchool/MachineLearning/Perceptron") 

#Instead of loading library tidyverse and doing read_csv, use built in R
#Load in provided data and label it as a data frame, there are headers
DirtyLaurino <- read.csv("ClassifyDirtyData_Laurino.csv", header = TRUE)
#Specify column headers or it will read them as data points.

#Retrieve the dimension of the dataframe
dim(DirtyLaurino)
#1000 obs of 3 variables
#By clicking the down arrow in the environment, you can see the data is separated by the headers, X1, X2 and Y

#Quick plot just to see what the data looks like:
with(DirtyLaurino, plot(X1, X2, col = Y+5, xlab="X Data",ylab="Y Data",main="Laurino Data"))
#Colors are altered with +5 to ensure results are not negative for visualization.

#Create training data: https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
training = sample(1:nrow(DirtyLaurino), size=0.8*nrow(DirtyLaurino))
#Take a sample by going through all of the rows in the dataset but only remember 80% of them - From powerpoint
trainingdata = DirtyLaurino[training,] #That comma is needed, stop deleting it
#Taking the sample from the dataset with specifications from training
dim(trainingdata)
#The training data contains 800 samples from the data with 1000 obs, currently containing all three columns.
rm(training) #Just to keep the environment clean

#Create test data: https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
test = sample(1:nrow(DirtyLaurino), size=0.2*nrow(DirtyLaurino))
#Take a sample by going through all of the rows in the dataset but only remember 20% of them - From powerpoint
testdata = DirtyLaurino[test,] 
#Taking the sample from the dataset with specifications from training
dim(testdata)
#The training data contains 200 samples from the data with 1000 obs.
rm(test) #Just to keep the environment clean

#Quick plots just to see what the data looks like
with(trainingdata, plot(X1,X2,col=Y+5,xlab="X Data",ylab="Y Data",main = "Laurino Training Data"))
with(testdata, plot(X1,X2,col=Y+5,xlab="X Data",ylab="Y Data",main = "Laurino Test Data"))

Y<- DirtyLaurino$Y #Third column/int

Y
#Looking at the column Y in the console displays values of -1 and 1, the two options for the percepton algorithm.
#This contains all ORIGINAL Y values for both training and test data.

#But, we need to assign a BIAS column for training and test data that has all of the same data, or same answer like true/yes
trainingY <- trainingdata$Y #800obs
testY <- testdata$Y #200obs

#Now that we have the Y data saved as other values, we can aknowledge them separately in the perceptron algorithm
#https://rpubs.com/FaiHas/197581
trainingdata  <- subset(trainingdata , select = -c(Y) )
testdata <- subset(testdata, select = -c(Y) )
#Should have only two variables in the environment now, this excludes the third column, Y
#Plus, we just saved the Y columns for all data, training and testing already anyway incase we need it.
#Add a third column for all the rows in the training/test data with the value 1 using replicate command,
#The rows have to contain the same number instead of a mixture so the accuracy/error can develop a percentage
#for if the function was right or not.
trainingdata$X3 <- rep(1, nrow(trainingdata)) #View in environment to ensure it worked
testdata$X3 <- rep(1, nrow(testdata)) #View in environment to ensure it worked
#Double check it in the console:
testdata$X3

#DEFINING THE VARIABLES NOT INSIDE THE FUNCTION WAS ALTERING MY ANSWERS - DO NOT DO THIS.
#Define variables needed for function:
#w1<-0 #Initial weight value
#w2<-0 #Weighted 2/Initial weight value
#w3<-0 #Weighted 3/Initial weight value
#rm(W1,W2,W3)
w <- c(0,0,0) #From powerpoint, another way to set weighted values to 0 instead of above^, we can use the values above for our error calculation
#X1<- DirtyLaurino$X1 #First column/num - Used for defining the function
#X2<- DirtyLaurino$X2 #Second column/num - Value names are from the default Dirty Laurino data.
#accuracy <- 0
#K<-0 #Initializer for counter
#L<-0
Kiterations <-150 #The number of times the function will repeat
#Jiterations <-0 #The counter

#Define the perceptron function: https://rpubs.com/FaiHas/197581 (Used just for classification of function)
perceptron <- function(x, y, w, Kiterations) { #Function of X data, Y data, the weights(3) and the number of times the function will repeat
    allclassified <- FALSE #The function does not work if you do not specify that allclassified is false first because it will say the object is not defined....
    accuracy <- 0
    K<-0 #Initializer for counter
    while (!allclassified) { #while they are not allclassified
      L<-0 #A separate counter for failures
      allclassified <- TRUE
      for (i in 1:nrow(x)) { #For every row in the data:
        w1 <- w[1]*x[i,1] #The new weight values will be the products using the weight results from the training data
        w2 <- w[2]*x[i,2]
        w3 <- w[3]*x[i,3]
        total <- w1+w2+w3 #The sum of all weight values will be the total
       
        if (sign(total) != y[i]) { #*From powerpoint* - sign returns a vector with the signs of the corresponding elements of x while they are not equal to y
        #if (sign(w %*% x[i,]) != y[i,]) Keeps giving error:Error in w %*% x[i, ] : requires numeric/complex matrix/vector arguments
          #allclassified = FALSE
          #ERROR: Product of vector w and row of x, != is not equal to, if the sign is not equal to the Y value, recalculate the weights:
          L <- L+1 #This new counter will keep track of the number of times it fails
          w1 <-w[1] + (x[i,1] * y[i]) #Product of all x&y plus the weight value for each value
          w2 <-w[2] + (x[i,2] * y[i])
          w3 <-w[3] + (x[i,3] * y[i])
          #w<-(c(w[1]+(x[i,1]*y[i]), w[2]+(x[i,2]*y[i]), w[3]+(x[i,3]*y[i])))#Changing all w values from 0 to their new products
          #w<-c(as.numeric(w1),as.numeric(w2),as.numeric(w3))
          w <- c(w1, w2, w3) #reclassify w with all weighted values
          allclassified  <-  FALSE #Do not end it yet
        }#Ends IF
      }#Ends FOR
      
      accuracy <- 100 - ((L / nrow(x)) * 100) #subtract error (the number of failures divided by the number of rows in the dataset times 100) from 100
      K <- K+1 #LABELING THIS OUT OF THE FUNCTION DOES NOT WORK
      if (K > Kiterations){ #If K ends up greater than the number of max iterations, calculate slope and intercept
        m <- -(w[1]/w[2]) #Slope equation from powerpoint
        b <- -(w[3]/w[2]) #Intercept equation from powerpoint, W3 is the BIAS 
        abline(b,m,col=1) #Drawing the line, ?abline is just drawing straight line on plot
        allclassified <- TRUE #STOP
      }#Ends 2nd IF
    }#Ends WHILE
    
  #Return the 3 weight results and accuracy
  return ( c(w, accuracy))

}#Ends FUNCTION

error <- function(x, y, w){
  Jiterations <- 0   #Counter for accurate classifications
    for (i in 1:nrow(x)) { #Loop for every row in the test data:
      w1 <- w[1]*x[i,1] #The new weight values will be the products using the weight results from the training data
      w2 <- w[2]*x[i,2]
      w3 <- w[3]*x[i,3]
      total <- w1+w2+w3 #The sum of all weight values
      if (sign(total) == y[i]){  #If the sign of the sum of the weights is equal, start the iterations
        Jiterations <- Jiterations + 1
    }#Ends IF
  }#Ends For
  #at the end, calculate a final accuracy measurement
  accuracy <- (Jiterations/nrow(x)) * 100
  #the function returns the accuracy value
  return (accuracy)
}#ENDS ERROR

#Plot the training data to have in the viewer for when we draw the slope next:
with(trainingdata, plot(X1,X2,col=trainingY+5,xlab="X Data",ylab="Y Data",main = "Laurino Training Data"))

#Compute the perceptron on the training results with the 4 requirements, x y w and Kiterations
TRAININGresults <- perceptron(trainingdata, trainingY, w, Kiterations)

TRAININGresults #The resulting weights printed in console
#10.920430  8.552384  1.000000 96.500000 - 150 iterations
#When I took a sample/subset of the data, it was random, so the results may vary slightly with the 
#different points the subset chooses, if you start over and run everything again.

#Draw the slope of the line on the graph of training data
#The negative sign has to be on the outside of the perenthesis 
m <- -(TRAININGresults[1]/TRAININGresults[2]) #Slope equation from powerpoint
b <- -(TRAININGresults[3]/TRAININGresults[2]) #Intercept equation from powerpoint, W3 is the BIAS 
abline(b,m,col=1) #Draw line of slope in black
m
#-1.276887 - 150 iterations
b
#-0.1169265 - 150 iterations

TESTresults <- error(testdata, testY, TRAININGresults) #Does not need Kiterations because it was not specified and the function has it's own iterations within it

#the out-of-training accuracy
TESTresults
#98.5

#
#The test results (98.5) are greater than the training results (96.5) which is expected because the
#perceptron was using the training data originally. The % error calculated for both the training and 
#test data is relatively low. The error will never be much smaller than the results below because if
#you look at the graph of the training data, there will always be overlapping color points on each side
#of the slope line.
#

# Express the in-training error as a percentage
trainingerror <- 100 - TRAININGresults[4] #(100-96.5)
trainingerror
#The error is 3.5%, which is relatively low, which is good

#Express the test error as a percentage
testerror <- 100 - TESTresults #(100-98.5)
testerror
#The error is 1.5%, which is relatively low, which is good




#OLD  while (K < 100) {
#OLD    if sign(w %*% x[i,]) != y[i]) #*From powerpoint*sign returns a vector with the signs of the corresponding elements of x 
#OLD      for (i in 1:nrow(trainingdata)) {
#OLD        W0 <- (W0 + Y[i]) 
#OLD        W1 <- W1 + (X1[i]*Y[i])
#OLD        W2 <- W2 + (X2[i]*Y[i]) 
#OLD    if(K > 1000) {
#OLD      list (W0, W1, W2) 