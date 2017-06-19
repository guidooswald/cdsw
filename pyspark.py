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



# # K-Means
#
# The K-means algorithm written from scratch against PySpark. In practice,
# one may prefer to use the KMeans algorithm in ML, as shown in
# [this example](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/kmeans_example.py).
# 
# This example requires [NumPy](http://www.numpy.org/).

from __future__ import print_function
import sys
import numpy as np
from pyspark.sql import SparkSession

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

spark = SparkSession\
    .builder\
    .appName("PythonKMeans")\
    .getOrCreate()

# Add the data file to hdfs.
!hdfs dfs -put resources/data/mllib/kmeans_data.txt /tmp

lines = spark.read.text("/tmp/kmeans_data.txt").rdd.map(lambda r: r[0])
data = lines.map(parseVector).cache()
K = 2
convergeDist = 0.1

kPoints = data.takeSample(False, K, 1)
tempDist = 1.0

while tempDist > convergeDist:
    closest = data.map(
        lambda p: (closestPoint(p, kPoints), (p, 1)))
    pointStats = closest.reduceByKey(
        lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
    newPoints = pointStats.map(
        lambda st: (st[0], st[1][0] / st[1][1])).collect()

    tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

    for (iK, p) in newPoints:
        kPoints[iK] = p

print("Final centers: " + str(kPoints))

spark.stop()



# # Alternating Least Squares
# 
# This is an example implementation of ALS for learning how to use Spark.
# Please refer to
# [this example](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/als_example.py) # for more conventional use.
# 
# This example requires [NumPy](http://www.numpy.org/)

from __future__ import print_function
import sys
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

LAMBDA = 0.01   # regularization
np.random.seed(42)

def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))

def update(i, vec, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)

spark = SparkSession\
    .builder\
    .appName("PythonALS")\
    .getOrCreate()

sc = spark.sparkContext

M = 100
U = 500
F = 10
ITERATIONS = 5
partitions = 2

print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
      (M, U, F, ITERATIONS, partitions))

R = matrix(rand(M, F)) * matrix(rand(U, F).T)
ms = matrix(rand(M, F))
us = matrix(rand(U, F))

Rb = sc.broadcast(R)
msb = sc.broadcast(ms)
usb = sc.broadcast(us)

for i in range(ITERATIONS):
    ms = sc.parallelize(range(M), partitions) \
           .map(lambda x: update(x, msb.value[x, :], usb.value, Rb.value)) \
           .collect()
    # collect() returns a list, so array ends up being
    # a 3-d array, we take the first 2 dims for the matrix
    ms = matrix(np.array(ms)[:, :, 0])
    msb = sc.broadcast(ms)

    us = sc.parallelize(range(U), partitions) \
           .map(lambda x: update(x, usb.value[x, :], msb.value, Rb.value.T)) \
           .collect()
    us = matrix(np.array(us)[:, :, 0])
    usb = sc.broadcast(us)

    error = rmse(R, ms, us)
    print("Iteration %d:" % i)
    print("\nRMSE: %5.4f\n" % error)

spark.stop()
