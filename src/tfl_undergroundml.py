# =======================
# IMPORT LIBRARIES
# =======================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, year, regexp_replace
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.linalg import Vectors

# =======================
# CREATE SPARK SESSION WITH HIVE SUPPORT
# =======================
spark = SparkSession.builder \
    .appName("Hive_Spark_Classification") \
    .enableHiveSupport() \
    .getOrCreate()

# =======================
# READ DATA FROM EXISTING HIVE TABLE
# =======================
df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
df.show(6)

# =======================
# FEATURE ENGINEERING
# =======================
df = df.withColumn('hour', hour(col('timestamp')))
df = df.withColumn('day_of_week', dayofweek(col('timestamp')))
df = df.withColumn('month', month(col('timestamp')))
df = df.withColumn('year', year(col('timestamp')))
df = df.withColumn('reason', regexp_replace(col('reason'), '[^a-zA-Z0-9 ]', ''))

# =======================
# STRING INDEXING
# =======================
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep").fit(df)
    for col in ["line", "reason", "route", "status"]
]
for indexer in indexers:
    df = indexer.transform(df)

# =======================
# ONE-HOT ENCODING
# =======================
encoder = OneHotEncoder(
    inputCols=["line_index", "reason_index", "route_index"],
    outputCols=["line_vec", "reason_vec", "route_vec"]
)
df = encoder.fit(df).transform(df)

# =======================
# VECTOR ASSEMBLER
# =======================
assembler = VectorAssembler(
    inputCols=["hour", "day_of_week", "month", "year", "line_vec", "reason_vec", "route_vec"],
    outputCol="features"
)
data = assembler.transform(df).select("features", "status_index")

# =======================
# TRAIN-TEST SPLIT
# =======================
train_data, test_data = data.randomSplit([0.8, 0.2], seed=123)

# =======================
# MODEL TRAINING
# =======================
# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="status_index")
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="status_index", numTrees=50)
rf_model = rf.fit(train_data)
rf_preds = rf_model.transform(test_data)

# =======================
# CREATE NEW HIVE TABLE AND STORE THE RESULTS
# =======================
# Create new table to store logistic regression predictions
lr_preds.createOrReplaceTempView("lr_preds_temp")
spark.sql("""
    CREATE TABLE IF NOT EXISTS tfl_underground_lr_predictions AS
    SELECT * FROM lr_preds_temp
""")

# Create new table to store random forest predictions
rf_preds.createOrReplaceTempView("rf_preds_temp")
spark.sql("""
    CREATE TABLE IF NOT EXISTS big_datajan2025.tfl_underground_rf_predictions AS
    SELECT * FROM rf_preds_temp
""")

# =======================
# STOP SPARK SESSION
# =======================
spark.stop()
