import sys
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType


class SSBFS:
    def __init__(self, app_name="SS-BFS", master="local[*]"):
        self.spark = SparkSession.builder \
            .appName("SS-BFS") \
            .master(master) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.sparkContext.setCheckpointDir("/tmp/graph_ckpt")

    def stop(self):
        self.spark.stop()

    def load_edges(self, path):
        """
        Загрузка ребёр из текстового файла: два целых номера на строку.
        """
        df = self.spark.read.text(path) \
            .select(F.split(F.col("value"), r"\s+").alias("cols")) \
            .filter(F.size("cols") == 2) \
            .select(
                F.col("cols").getItem(0).cast("long").alias("src"),
                F.col("cols").getItem(1).cast("long").alias("dst")
            )
        rev = df.select(F.col("dst").alias("src"), F.col("src").alias("dst"))
        edges = df.union(rev).distinct() \
            .repartition(200, "src") \
            .persist(StorageLevel.MEMORY_ONLY)
        return edges

    def run_ss_bfs(self, edges_df, source):
        """
        Single-Source BFS: source — стартовая вершина (одна).
        Вернёт DataFrame (vertex, distance, parent).
        """
        schema = StructType([
            StructField("vertex", IntegerType(), nullable=False),
            StructField("dist", IntegerType(), nullable=False),
            StructField("parent", IntegerType(), nullable=True)
        ])

        src_df = self.spark.createDataFrame(
            [(int(source), 0, None)],
            schema=schema
        )

        frontier = src_df.persist(StorageLevel.MEMORY_ONLY)
        visited = src_df.persist(StorageLevel.MEMORY_ONLY)

        frontier = frontier.checkpoint()
        visited = visited.checkpoint()

        while True:
            nbrs = frontier.alias("f") \
                .join(edges_df.alias("e"), F.col("f.vertex") == F.col("e.src")) \
                .select(
                    F.col("e.dst").alias("vertex"),
                    (F.col("f.dist") + 1).alias("dist"),
                    F.col("f.vertex").alias("parent")
                ) \
                .distinct()

            new_front = nbrs.join(
                visited.select("vertex"),
                on="vertex",
                how="left_anti"
            ).persist(StorageLevel.MEMORY_ONLY)

            if new_front.rdd.isEmpty():
                break

            visited = visited.union(new_front).persist(StorageLevel.MEMORY_ONLY)
            frontier = new_front

            visited = visited.checkpoint()

        return visited

def main():
    if len(sys.argv) < 3:
        print("Usage: ss_bfs.py <edge_file> <start_vertex>")
        sys.exit(1)

    path = sys.argv[1]
    source = sys.argv[2]

    bfs = SSBFS()
    edges = bfs.load_edges(path)
    result = bfs.run_ss_bfs(edges, source)

    result.orderBy("dist").show(20, truncate=False)
    print(f"Total reached: {result.count()}")

    bfs.stop()

if __name__ == "__main__":
    main()
