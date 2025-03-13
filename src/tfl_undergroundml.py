from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, year, regexp_replace
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

# Create Spark session with Hive support
spark = SparkSession.builder \
    .appName("Hive_Spark_Classification") \
    .enableHiveSupport() \
    .getOrCreate()

# Read data from the Hive table
df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
df.show(6)

# Ensure no empty or invalid column names
df = df.select([col(c).alias(c if c else "valid_column_name") for c in df.columns])

# Feature Engineering
df = df.withColumn('hour', hour(col('timestamp')))
df = df.withColumn('day_of_week', dayofweek(col('timestamp')))
df = df.withColumn('month', month(col('timestamp')))
df = df.withColumn('year', year(col('timestamp')))
df = df.withColumn('reason', regexp_replace(col('reason'), '[^a-zA-Z0-9 ]', ''))

# String Indexing
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="skip").fit(df)
    for col in ["line", "reason", "route", "status"]
]
for indexer in indexers:
    df = indexer.transform(df)

# One-Hot Encoding for each column
encoder_line = OneHotEncoder(inputCol="line_index", outputCol="line_vec")
encoder_reason = OneHotEncoder(inputCol="reason_index", outputCol="reason_vec")
encoder_route = OneHotEncoder(inputCol="route_index", outputCol="route_vec")

df = encoder_line.fit(df).transform(df)
df = encoder_reason.fit(df).transform(df)
df = encoder_route.fit(df).transform(df)

# Vector Assembler
assembler = VectorAssembler(
    inputCols=["hour", "day_of_week", "month", "year", "line_vec", "reason_vec", "route_vec"],
    outputCol="features"
)
data = assembler.transform(df).select("features", "status_index")

# Train-test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=123)

# Model Training
# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="status_index")
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="status_index", numTrees=50)
rf_model = rf.fit(train_data)
rf_preds = rf_model.transform(test_data)

# Create new Hive tables to store predictions
lr_preds.createOrReplaceTempView("lr_preds_temp")
spark.sql("""
    CREATE TABLE IF NOT EXISTS tfl_underground_lr_predictions AS
    SELECT * FROM lr_preds_temp
""")

rf_preds.createOrReplaceTempView("rf_preds_temp")
spark.sql("""
    CREATE TABLE IF NOT EXISTS big_datajan2025.tfl_underground_rf_predictions AS
    SELECT * FROM rf_preds_temp
""")

# Stop Spark session
spark.stop()
