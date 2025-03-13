# =======================
# IMPORT LIBRARIES
# =======================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, year, regexp_replace
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

# =======================
# CREATE SPARK SESSION WITH HIVE SUPPORT
# =======================
spark = SparkSession.builder \
    .appName("Hive_Spark_Classification") \
    .enableHiveSupport() \
    .getOrCreate()

# =======================
# READ DATA FROM HIVE TABLE
# =======================
df = spark.sql("SELECT * FROM scala_tfl_underground")
df.show(6)

# =======================
# FEATURE ENGINEERING
# =======================
# Extract time-based features
df = df.withColumn('hour', hour(col('timestamp')))
df = df.withColumn('day_of_week', dayofweek(col('timestamp')))
df = df.withColumn('month', month(col('timestamp')))
df = df.withColumn('year', year(col('timestamp')))

# Clean 'reason' column (remove special characters)
df = df.withColumn('reason', regexp_replace(col('reason'), '[^a-zA-Z0-9 ]', ''))

# =======================
# STRING INDEXING (Convert categorical to numeric)
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
# VECTOR ASSEMBLER (Combine all features)
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
# MODEL TRAINING AND PREDICTION
# =======================

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(featuresCol="features", labelCol="status_index")
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# Save Logistic Regression Predictions to Hive
lr_preds.write.mode("overwrite").saveAsTable("scala_tfl_lr_predictions")

# Save Logistic Regression Model to Hive
lr_model.save("hdfs:///models/lr_model")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(featuresCol="features", labelCol="status_index", numTrees=50)
rf_model = rf.fit(train_data)
rf_preds = rf_model.transform(test_data)

# Save Random Forest Predictions to Hive
rf_preds.write.mode("overwrite").saveAsTable("scala_tfl_rf_predictions")

# Save Random Forest Model to Hive
rf_model.save("hdfs:///models/rf_model")

# =======================
# STOP SPARK SESSION
# =======================
spark.stop()
