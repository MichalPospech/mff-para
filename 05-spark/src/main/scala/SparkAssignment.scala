import org.apache.spark.sql.SparkSession
import java.io._

object SparkAssignment {
  def main(args: Array[String]): Unit = {
    val namesFile = args(0)
    val spark = SparkSession.builder().appName("Spark Assignment").getOrCreate()
    import spark.implicits._
    val df = spark.read
      .csv(namesFile)
      .toDF("firstName", "lastName", "telNumber", "zipCode")
    val simpleDf =
      df.withColumn("region", $"zipCode" substr (0, 1))
    val grouped = simpleDf.groupBy($"region", $"firstName", $"lastName")
    val namesInRegions = grouped.count().filter($"count" > 1)
    val result = namesInRegions.groupBy($"region").agg(Map("count" -> "sum")).orderBy($"region").collect()
    var outCsv = new PrintWriter(new File(args(1)))
    result.foreach(
      r => outCsv.println(r.mkString(","))
    )
    outCsv.close()
    spark.stop()
  }
}
