//package org.apache.spark.examples
package com.tencent.tdw.spark.interview

import scala.collection.mutable
import breeze.linalg._
import org.apache.spark._
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by kingsfield on 2016/9/4.
  */
object Localization {

  val iteration=50
  val fakeZero=0.000001
  /*
  find the point where sum of all Euclidean distances to the training points is minimum.
  we use Weiszfeld's algorithm to avoid calculate gradients.
  https://en.wikipedia.org/wiki/Geometric_median
   */
  def bestL1Predict(points: Array[DenseVector[Double]], iter: Int): DenseVector[Double] = {
    val smallDouble = 0.01
    var ret= new DenseVector(points(0).toArray)
    for (i <- 1 to iter) {
      var x = DenseVector.zeros[Double](2)
      var sumWeight=0.0
      for (p <- points) {
        val diff = p-ret
        val weight=1.9/(diff.length+smallDouble)
        x+=weight*p
        sumWeight+=weight
      }
      x/=sumWeight
      ret=x
    }
    return ret
  }

  def parseNdPoints(raw: String, n: Int): Array[DenseVector[Double]] = {
    val split=raw.split(",")
    val size = (split.length/n).toInt
    val ret = new Array[DenseVector[Double]](size)
    for (i<-0 to size) {
      val v=DenseVector.zeros[Double](n)
      for (k<-0 to n) {
        v(k)=split(i*n+k).toDouble
      }
      ret(i)=v
    }
    return ret
  }

  def oldTrainOffline(sc:SparkContext): Unit = {
    // raw is training data which is 'mac_i:x,y' each line
    val raw = sc.textFile("hdfs://path")
    val kv = raw.map(line=>{
      val ss = line.split(";")
      (ss(0),parseNdPoints(ss(1),2)(0))
    })
    val training = kv.groupByKey()
    val model = training.map(row=>{
      (row._1,bestL1Predict(row._2.toArray,iteration))
    })
    model.saveAsTextFile("hdfs://path")
  }

  def parseValidation2d(raw: String, model : mutable.HashMap[String,DenseVector[Double]]): Tuple2[DenseVector[Double],mutable.ArrayBuffer[DenseVector[Double]]] = {
    val split=raw.split(",")
    val real=DenseVector(split(0).toDouble,split(1).toDouble)
    val macLocations=mutable.ArrayBuffer[DenseVector[Double]]()
    for (i <- 2 to split.length) {
      val mac = split(i)
      if (model.contains(mac)) {
        macLocations.append(model(mac))
      }
    }
    return (real,macLocations)
  }

  /*
  assume we have already got all mac model which is mac->(x,y)
   */
  def oldValidationOffline(sc:SparkContext, model : mutable.HashMap[String,DenseVector[Double]]): Unit = {
    // raw is localization context, which is 'real_x,real_y,mac1,mac2,mac3,...' each line
    val raw = sc.textFile("hdfs://path")
    val broadcastModel = sc.broadcast(model)
    val validation = raw.map(line=>{
      parseValidation2d(line,broadcastModel.value)
    }).map(row=>{
      (row._1,bestL1Predict(row._2.toArray,iteration))
    }).map(row=>{
      val diff = row._1-row._2
      norm(diff)
    }).collect()
    // then calculate the result off validation
  }

  def new1000kMillionDollarsMethodTrain(sc:SparkContext): Unit = {
    // raw is training data which is 'mac_i:x,y,signal' each line
    val raw = sc.textFile("hdfs://path")
    val kv = raw.map(line=>{
      val ss = line.split(";")
      (ss(0),parseNdPoints(ss(1),3)(0))
    })
    val training = kv.groupByKey()
    val model = training.map(row=>{
      // train a gmm locally
      // we omit training process for simplicity which it's easy to code a locally trained gmm model
      // val model=new GaussianMixtureModel()
    })
    model.saveAsTextFile("hdfs://path")
  }

  def gmmProb(model: GaussianMixtureModel, point: DenseVector[Double]): Double = {
    val p = model.weights.zip(model.gaussians).map {
      case (weight, dist) => fakeZero + weight * dist.pdf(Vectors.dense(point.toArray))
    }
    return p.sum
  }

  def newPredict(macStr:String, allModel: mutable.HashMap[String,GaussianMixtureModel]): DenseVector[Double] = {
    val split = macStr.split(",")
    val models = new mutable.ArrayBuffer[GaussianMixtureModel]()
    for (mac <- split) {
      if (allModel.contains(mac)) {
        models.append(allModel(mac))
      }
    }
    val samplePoints = new mutable.ArrayBuffer[DenseVector[Double]]()
    for (model <- models) {
      for (gaussian <- model.gaussians) {
        samplePoints.append(DenseVector(gaussian.mu.toArray))
      }
    }
    var sumWeigth=0.0
    val ret = DenseVector.zeros[Double](2)
    // we use simple sampling strategy
    for (point <- samplePoints) {
      var weight = 1.0
      for (model <- models) {
        weight *= gmmProb(model, point)
      }
      sumWeigth+=weight
      ret+=point*weight
    }
    ret/=sumWeigth
    return ret
  }

  def new1000kMillionDollarsMethodValidation(sc:SparkContext, model: mutable.HashMap[String,GaussianMixtureModel]): Unit = {
    // raw is localization context, which is 'real_x,real_y,mac1,mac2,mac3,...' each line
    val raw = sc.textFile("hdfs://path")
    val broadcastModel = sc.broadcast(model)
    val validation = raw.map(line=>{
      val split = line.split(",")
      val real = DenseVector[Double](split(0).toDouble,split(1).toDouble)
      val sz = split.length-2
      val predict = newPredict(split.takeRight(sz).mkString(","),broadcastModel.value)
      (real,predict)
    }).map(row=>{
      val diff = row._1 - row._2
      norm(diff)
    }).collect()
    // then calculate the result off validation
  }

  def main(args: Array[String]): Unit = {
    val sparkConf_ = new SparkConf()
    sparkConf_.setAppName("InterviewShowcase")
    sparkConf_.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf_.set("spark.default.parallelism", "1000")
    sparkConf_.set("spark.storage.memoryFraction", "0.5")
    sparkConf_.set("spark.shuffle.memoryFraction", "0.3")
    sparkConf_.set("spark.shuffle.memoryFraction", "0.3")
    sparkConf_.set("spark.driver.memory", "16g")
    sparkConf_.set("spark.driver.maxResultSize", "12g")
    val sc = new SparkContext(sparkConf_)

    // old method
    oldTrainOffline(sc)
    // suppose we have load models from oldTrainOffline
    val oldModels = new mutable.HashMap[String,DenseVector[Double]] ()
    oldValidationOffline(sc,oldModels)

    // new method
    new1000kMillionDollarsMethodTrain(sc)
    // suppose we have load models from new1000kMillionDollarsMethodTrain
    val newModels = new mutable.HashMap[String,GaussianMixtureModel]()
    new1000kMillionDollarsMethodValidation(sc,newModels)
  }
}
