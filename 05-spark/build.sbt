lazy val root = project
  .in(file("."))
  .settings(
    name := "MFF Para - Spark Assignment",
    version := "0.1.0",
    scalaVersion := "2.12.13"
  )

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x                             => MergeStrategy.first
}

libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.1.1"
