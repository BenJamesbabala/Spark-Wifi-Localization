package org.apache.spark.examples

import scala.collection.mutable
import breeze.linalg.{squaredDistance, DenseVector, Vector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

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
  def bestL1Predict(points: mutable.ArrayBuffer[DenseVector[Double]], iter: Int): Unit = {
    val smallDouble = 0.01
    var ret=points(0)
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

  def parseNdPoints(raw: String, n: Int): mutable.ArrayBuffer[DenseVector[Double]] = {
    val ret = mutable.ArrayBuffer[DenseVector[Double]]()
    val split=raw.split(",")
    val size = (split.length/n).toInt
    for (i<-0 to size) {
      val v=DenseVector.zeros[Double](n)
      for (k<-0 to n) {
        v(k)=split(i*n+k).toDouble
      }
      ret.append(v)
    }
    return ret
  }

  def oldTrainOffline(): Unit = {
    val spark = SparkSession
      .builder
      .appName("oldTrainOffline")
      .getOrCreate()

    // raw is training data which is 'mac_i:x,y' each line
    val raw = spark.sparkContext.textFile("hdfs://path")
    val kv = raw.map(line=>{
      val ss = line.split(";")
      (ss(0),ss(1))
    })
    val training = kv.groupByKey().map(row=>{
      (row._1,parseNdPoints(row._2.toString(),2))
    })
    val model = training.map(row=>{
      (row._1,bestL1Predict(row._2,iteration))
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
  def oldValidationOffline(model : mutable.HashMap[String,DenseVector[Double]]): Unit = {
    val spark = SparkSession
      .builder
      .appName("oldTrainOffline")
      .getOrCreate()

    // raw is localization context, which is 'real_x,real_y,mac1,mac2,mac3,...' each line
    val raw = spark.sparkContext.textFile("hdfs://path")
    val validation = raw.map(line=>{
      parseValidation2d(line,model)
    }).map(row=>{
      (row._1,bestL1Predict(row._2,iteration))
    }).map(row=>{
//      val diff = row._1-row._2
//      diff.asInstanceOf[DenseVector].length
    }).collect()
    // then calculate the result off validation

  }

  def new1000kMillionDollarsMethodTrain(): Unit = {
    val spark = SparkSession
      .builder
      .appName("oldTrainOffline")
      .getOrCreate()

    // raw is training data which is 'mac_i:x,y,signal' each line
    val raw = spark.sparkContext.textFile("hdfs://path")
    val kv = raw.map(row=>{
      val split=row.split(":")
      (split(0),split(1))
    })
    val training = kv.groupByKey()

  }

  def gmmProb(model: GaussianMixtureModel, point: DenseVector[Double]): Double = {
    val p = model.weights.zip(model.gaussians).map {
      case (weight, dist) => fakeZero + weight * dist.pdf(Vectors.fromBreeze(point))
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
        samplePoints.append(gaussian.mu.asBreeze.asInstanceOf[DenseVector[Double]])
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

  def new1000kMillionDollarsMethodValidation( model: mutable.HashMap[String,GaussianMixtureModel]): Unit = {
    val spark = SparkSession
      .builder
      .appName("new1000kMillionDollarsMethodValidation")
      .getOrCreate()

    // raw is localization context, which is 'real_x,real_y,mac1,mac2,mac3,...' each line
    val raw = spark.sparkContext.textFile("hdfs://path")
    val validation = raw.map(line=>{
      val split = line.split(",")
      val real = DenseVector[Double](split(0).toDouble,split(1).toDouble)
      val sz = split.length-2
      val predict = newPredict(split.takeRight(sz).mkString(","),model)
      (real,predict)
    }).map(row=>{
      val diff = row._1 - row._2
      (diff.length)
    }).collect()
    // then calculate the result off validation
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("oldTrainOffline")
      .getOrCreate()
    var rdd1 = spark.sparkContext.makeRDD(Array(("A",0),("A",2),("B",1),("B",2),("C",1)))
  }
}
