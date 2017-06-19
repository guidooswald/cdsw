# # Estimating $\pi$
#
# This is the simplest PySpark example. It shows how to estimate $\pi$ in parallel
# using Monte Carlo integration. If you're new to PySpark, start here!

from __future__ import print_function
import sys
from random import random
from operator import add
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonPi")\
    .getOrCreate()

partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
n = 100000 * partitions

def f(_):
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 < 1 else 0

count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
print("Pi is roughly %f" % (4.0 * count / n))

spark.stop()



# # Sorting
#
# Sorting key-value pairs by key.

from __future__ import print_function
import sys
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonSort")\
    .getOrCreate()

# Add the data file to hdfs.
!hdfs dfs -put resources/kv1.txt /tmp

lines = spark.read.text("/tmp/kv1.txt").rdd.map(lambda r: r[0])
sortedCount = lines.flatMap(lambda x: x.split(' ')[0]) \
    .map(lambda x: (int(x), 1)) \
    .sortByKey()

# This is just a demo on how to bring all the sorted data back to a single node.
# In reality, we wouldn't want to collect all the data to the driver node.
output = sortedCount.collect()
for (num, unitcount) in output:
    print(num)

spark.stop()


# # Spark-SQL from PySpark
# 
# This example shows how to send SQL queries to Spark.

from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .getOrCreate()

# A list of Rows. Infer schema from the first row, create a DataFrame and print the schema
rows = [Row(name="John", age=19), Row(name="Smith", age=23), Row(name="Sarah", age=18)]
some_df = spark.createDataFrame(rows)
some_df.printSchema()

# A list of tuples
tuples = [("John", 19), ("Smith", 23), ("Sarah", 18)]

# Schema with two fields - person_name and person_age
schema = StructType([StructField("person_name", StringType(), False),
                    StructField("person_age", IntegerType(), False)])

# Create a DataFrame by applying the schema to the RDD and print the schema
another_df = spark.createDataFrame(tuples, schema)
another_df.printSchema()

# Add the data file to hdfs.
!hdfs dfs -put resources/people.json /tmp

# A JSON dataset is pointed to by path.
# The path can be either a single text file or a directory storing text files.
if len(sys.argv) < 2:
    path = "hdfs://" + \
        os.path.join(os.environ['SPARK_HOME'], "/tmp/people.json")
else:
    path = sys.argv[1]
# Create a DataFrame from the file(s) pointed to by path
people = spark.read.json(path)

# The inferred schema can be visualized using the printSchema() method.
people.printSchema()

# Creates a temporary view using the DataFrame.
people.createOrReplaceTempView("people")

# SQL statements can be run by using the sql methods provided by `spark`
teenagers = spark.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")

for each in teenagers.collect():
    print(each[0])

spark.stop()



# # Word counts
# 
# This example shows how to count the occurrences of each word in a text file.

from __future__ import print_function
import sys, re
from operator import add
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonWordCount")\
    .getOrCreate()

# Add the data file to hdfs.
!hdfs dfs -put resources/cgroup-v2.txt /tmp

lines = spark.read.text("/tmp/cgroup-v2.txt").rdd.map(lambda r: r[0])
counts = lines.flatMap(lambda x: x.split(' ')) \
              .map(lambda x: (x, 1)) \
              .reduceByKey(add) \
              .sortBy(lambda x: x[1], False)
output = counts.collect()
for (word, count) in output:
    print("%s: %i" % (word, count))

spark.stop()
