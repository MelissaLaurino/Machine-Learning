#Melissa Laurino
#Machine Learning - KMeans Project 1
#2/1/17

#-Write R code for a K-Means Clustering algorithm using Euclidean distances from scratch, 
#--which prints the total within-cluster sum of squares and 
#--displays a graph using the Elbow Method to determine an appropriate K.

#Document each step of the code to demonstrate you understand what each line of code does. 
#The code has to describe the steps in creating the model and steps in computing the 
#sum of square errors.

#For your algorithm, you may use basic statistical functions and graphing functions but 
#NOT machine learning functions such as kmeans(). 

#Feel free to experiment with alternative measurements but at least one method 
#must employ Euclidean distances.

#Set working directory:
setwd("/Users/Melissa/Documents/GradSchool/MachineLearning/KNN") 

#Instead of loading library tidyverse and doing read_csv, use built in R
#Load in provided data and label it as a data frame, there are no headers
Laurino <- read.csv("KMeansData_Laurino.csv", header = FALSE)

#Retrieve the dimension of the dataframe
dim(Laurino)
#659 rows with 2 columns; Looks correct

head(Laurino)
#Use the head command to view the data in the data frame, double check it, 
#and also supplies V1 and V2 as the column "names" when we need them for graphing.

K=4
centers=Laurino[sample.int(nrow(Laurino),K),]
#The number of centers, K, will change how the algorithm behaves
#It is taking 5 random starting points (X,Y) ordered pair and storing them in a vector/list.
#These will eventually turn into the centers.
#For now, I am guessing 5 based off of the assignment instructions, but this can be changed
#based off the the elbow method graph for finding clusters after the kMeans from scratch.

current_stop_time=10e10
#This will evaluate the maximum distance away

cluster=rep(0,nrow(Laurino))
#rep will replicate the values of x, it takes the empty cluster in the environment for now
#and assigns it a bunch of zeros that will later be replaced and saved
#with the (X,Y) coordinates for the clusters.
#You can see this if you look at the environment ->
#It takes 659 because we tell it to take the number of rows in the Laurino dataframe.

converged=F
#F will be false and will be stopped when it turns to true.
#This will be used later to stop the loop.

it=1
#To be used for iterations, will be used later for incrementing through
#the dataset. For now, it starts at 1.

stop_time=10e-5
#We will eventually have to tell the code when to stop so it does not go on forever in the loop.
#This is the minimum distance the points can stop at, a very small number.

#We have our variables that we need to complete the equation, now we need to code it:

#Creating the function manually
kmeansLaurino=function(Laurino)
{
  while(current_stop_time>=stop_time & converged==F) #Start LOOP: While the current stop time is greater than 
                  #or equal to the stop time, and converged is still False, perform the following:
  {
    it=it+1 #Make the iteration equal to itself+1 to initiate a progression
    if (current_stop_time<=stop_time) #IF instace: If the current stop time is less than or equal to the stop time,
    {converged=T} #...then converged will change to True.
    old_centers=centers #...and old centers is now equal to centers
    #Now we have to assign points to the empty clusters we created first in the code:
    for (i in 1:nrow(Laurino)) #For every row in the Laurino dataset...
    {
      min_dist=10e10 #Keeping the minimum distance the same as the current stop..
      #..and as very large number. We now have to calculate Euclidean Distance; = sqrt[(x1-Ax1)^2 + (y1-Ay1)^2]:
      for (center in 1:nrow(centers)) #For every center in the centers value:
      {
        distance_to_center=sum((centers[center,]-Laurino[i,])^2) #The distnace to the center is equal to
        #the sum of the center coordinates minus the length of the data, squared.
        if (distance_to_center<=min_dist)# If the distnace to the center is less than or equal to the minimum distnace...
        #This centroid is the closest centroid to the point
        {cluster[i]=center #...than the point will be assigned to the cluster it is computing.
          min_dist=distance_to_center}
      } #Closes calculation of Euclidean Distance
    } #Closes Euclidean Distance
    #Also inside the loop: This is needed to calculate the distance, it will continue to calculate the distance for all clusters/centers
    for (i in 1:nrow(centers)) #For every center that was predefined..
    {centers[i,]=apply(Laurino[cluster==i,],2,mean)} #..apply each cluster in the Laurino dataset
    current_stop_time=mean((old_centers-centers)^2) #The current stop time is equal to the mean of the real centers squared, creates new stop limit
  } #Closes the details of the Loop.
  return(list(Laurino=data.frame(Laurino,cluster),centers=centers)) #Return a list of the Laurino data frame as it's own list that shows the clusters and the centers of the clusters.
} #Closes kmeans() from scratch.

##########
#Graphing#
##########
#Graphing the kmeansLaurino function results that we just obtained:

plot(Laurino, main = "Laurino Data Set", pch =1, cex =1)
#Just a regular plot of the data.

library(ggplot2)
#It is okay to use ggplot because it is a graphing library
resultsgraph=kmeansLaurino(Laurino[1:2])
resultsgraph #STOP. These take a while to print in the console.
#The results that are printed in the console display the value results graph 
#as a list of 2 dataframes, Laurino and centers:
#Columns that contain X and Y values are called V1 and V2
#The center coordinates are indicated as resultsgraph$centers
#The assigned cluster for the coordinates are within V3 labeled "cluster" or Laurino$cluster
resultsgraph$centers$cluster=1:4 #There are 4 clusters
resultsgraph$Laurino$iscenter=F #Coordinates that are not the true center
resultsgraph$centers$iscenter=T #Coordinates at the center
alldata=rbind(resultsgraph$centers,resultsgraph$Laurino) #Take the sequences and combine the data
ggplot(alldata,aes(x=V1,y=V2,color=as.factor(cluster),size=iscenter,alpha=iscenter))+geom_point()+labs(title = "K Means Cluster Results From Scratch - Laurino")
#The KMeans algorithm is used to find groups in the data set that are not labeled as groups already.

#############
#Elbow Graph#
#############

#K Means Clustering minimizes the WSS (Within-Cluster Sum of Squares)
#The Elbow graphing method using WSS will show us the optimal number of clusters or K to use:
#The mean of all the distances within the cluster.
#In order for this to work you have to CLEAR envivronment and console and only run the data (Laurino) 
#at the beginning because I manually label the number of centers for K at the beginning based off the results of the elbow graph.
wss <- (nrow(Laurino)-1)*sum(apply(Laurino,2,var))
for (i in 2:10) wss[i] <- sum(kmeans(Laurino,centers=i)$withinss) #I have to work this into my code instead and get rid of kmeans()
plot(1:10, wss, type="b", xlab="Number of Clusters",
                          ylab="Sum of Squares per Group",
                          main="Optimal Number of Clusters using Elbow Method",pch=20, cex=2)
#After creating an Elbow Method graph, 4 is the correct amount of clusters.
#the amount of clusters is shown where there is a significant "kink" in the plotted values
#before it becomes relatively consistent. 
#This should be done before creating the kMeans() code from scratch.

######################################
#Comparing to R's Version of KMeans()#
######################################

Rversion = kmeans(Laurino, 4)
Rversion
#Print Rversion of the kMeans() algorithm will show you the compenents of the data (cluster, centers, etc.),
#the WSS, the cluster  means, and the cluster sizes.

Rversion_graph <- kmeans(Laurino, centers = 4)
Rversion_graph
plot(Rversion_graph$centers,main="R Version KMeans()",xlab = "X Values",ylab = "Y Values")
#Use the centers component that was printed in the console from R's kMeans()
#Next, we want to see the clusters and the center points:
plot(Laurino, col =(Rversion_graph$cluster),
     main="R Version kMeans() - Laurino", pch=1, cex=1,
     xlab = "X Values",
     ylab = "Y Values")
points(Rversion_graph$centers, pch=18, cex=2)
#?pch #Choose shape of points (pch=#), ?cex is the size




######################################################
#DOES NOT WORK v

list(Laurino=data.frame(Laurino,cluster),centers=centers)
#centers
#       V1        V2
#570 10.330187 1.0597688
#616  6.251203 6.7017062
#281  9.670456 0.7719141
#548  6.200259 6.7423938
#277  9.342927 0.9066844
#452  1.729853 1.3116634
results <- list(Laurino=data.frame(Laurino,cluster),centers=centers)
results
results_data <-results$Laurino
results_centroid <-results$centers

