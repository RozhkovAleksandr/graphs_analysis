import sys
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, LongType

class SSSP:
    def __init__(self, app_name="Ford-Bellman", master="local[*]"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.sparkContext.setCheckpointDir("/tmp/graph_ckpt")
        self.INF = float('inf')  # Используем бесконечность для float

    def stop(self):
        self.spark.stop()

    def load_edges(self, path):
        return self.spark.read.text(path) \
            .select(F.split(F.col("value"), r"\s+").alias("cols")) \
            .filter(F.size("cols") >= 2) \
            .select(
                F.col("cols").getItem(0).cast("long").alias("src"),
                F.col("cols").getItem(1).cast("long").alias("dst"),
                F.when(F.size("cols") >= 3, 
                      F.col("cols").getItem(2).cast("float")).otherwise(1.0).alias("weight")
            ).persist(StorageLevel.MEMORY_ONLY)

    def run_sssp(self, edges_df, source):
        edges = edges_df.select(
            F.col("src").alias("from_vertex"),
            F.col("dst").alias("to_vertex"),
            "weight"
        ).persist(StorageLevel.MEMORY_AND_DISK)
        
        vertices = edges.select("from_vertex").union(edges.select("to_vertex")).distinct()
        distances = vertices.select(
            F.col("from_vertex").alias("vertex"),
            F.when(F.col("from_vertex") == source, 0.0).otherwise(self.INF).alias("distance")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        
        for _ in range(vertices.count() - 1):

            updated_distances = edges.join(
                distances.withColumnRenamed("distance", "current_distance"),
                edges["from_vertex"] == distances["vertex"]
            ).select(
                F.col("to_vertex").alias("vertex"),
                (F.col("current_distance") + F.col("weight")).alias("new_distance")
            )
            
            min_distances = updated_distances.groupBy("vertex").agg(
                F.min("new_distance").alias("min_distance")
            )
            
            new_state = distances.join(
                min_distances, "vertex", "left"
            ).select(
                "vertex",
                F.coalesce(
                    F.when(F.col("min_distance") < F.col("distance"), F.col("min_distance")),
                    F.col("distance")
                ).alias("distance")
            ).persist(StorageLevel.MEMORY_AND_DISK)
            
            if distances.exceptAll(new_state).isEmpty():
                distances.unpersist()
                break
                
            distances.unpersist()
            distances = new_state
        
        edges.unpersist()
        return distances.orderBy("vertex")

def main():
    if len(sys.argv) < 3:
        print("Usage: sssp.py <edge_file> <start_vertex>")
        sys.exit(1)

    path = sys.argv[1]
    source = int(sys.argv[2])

    sssp = SSSP()

    edges = sssp.load_edges(path)
    result = sssp.run_sssp(edges, source)
    
    print(f"Shortest paths from vertex {source}:")
    result.show(20, truncate=False)

    sssp.stop()

if __name__ == "__main__":
    main()