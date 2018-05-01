// Databricks notebook source
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
// load and parse the data
val ratingsFile = sc.textFile("/FileStore/tables/ratings.dat")
val timeOmit=ratingsFile.map(_.split("::")match{case Array(userid,movieid,ratings,timestamp)=>Rating(userid.toInt, movieid.toInt, ratings.toDouble)})
//split the data for 60% training and 40% testing
val splitArray = timeOmit.randomSplit(Array(0.6,0.4))
val trainingData = splitArray(0)
val testingData = splitArray(1)
//Build the recommendation model using ALS
val rank = 10
val numIterations = 30
val lambda = 0.01
val model = ALS.train(trainingData,rank,numIterations,lambda)

val userMovie = testingData.map{ case Rating(userid,movieid,rating)=>(userid,movieid)}
val actualRating = testingData.map { case Rating(userid, movieid, rating)=>((userid,movieid),rating) }
val predictedRating = model.predict(userMovie).map { case Rating(userid,movieid,rating)=>((userid,movieid),rating)}
val bothRatings = actualRating.join(predictedRating)
val MSE = bothRatings.map(x=>(math.pow((x._2._1 - x._2._2),2))).mean()
println("mean squared error is "+MSE)


// COMMAND ----------


