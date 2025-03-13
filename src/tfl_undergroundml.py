from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------

spark = SparkSession.builder.appName("TFL_Underground_Delay_Prediction").enableHiveSupport().getOrCreate()

# ---------------------------
# 2. Load Data from Hive
# ---------------------------

hive_df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
hive_df.show(5)

# Data Cleaning: Handle missing values
hive_df = hive_df.fillna({
    'scala_tfl_underground.line': 'Unknown',
    'scala_tfl_underground.route': 'Unknown',
    'scala_tfl_underground.delay_time': '0'
})

# Create a binary label column: 1 if there is a delay, 0 otherwise
hive_df = hive_df.withColumn("is_delayed", when(col("scala_tfl_underground.status").contains("Delay"), 1).otherwise(0))

# Check dataset balance
data = hive_df.groupBy('is_delayed').count()
data.show()

# ---------------------------
# 3. Feature Engineering
# ---------------------------

# Index categorical columns
line_indexer = StringIndexer(inputCol='line', outputCol='line_index')
route_indexer = StringIndexer(inputCol='route', outputCol='route_index')

# Assemble features
assembler = VectorAssembler(inputCols=['line_index', 'route_index'], outputCol='features')

# ---------------------------
# 4. Build and Train Model
# ---------------------------

print("Training the model...")

# Prepare data
prepared_df = line_indexer.fit(hive_df).transform(hive_df)
prepared_df = route_indexer.fit(prepared_df).transform(prepared_df)
prepared_df = assembler.transform(prepared_df)

# Split data into training and testing sets
train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
print("Training Set: {} rows, Testing Set: {} rows".format(train_df.count(), test_df.count()))

# Logistic Regression Model
lr = LogisticRegression(featuresCol='features', labelCol='is_delayed', maxIter=10)

model = lr.fit(train_df)

# Evaluate Model
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol='is_delayed')
auc = evaluator.evaluate(predictions)
print("Model AUC: {}".format(auc))

# Display Predictions
predictions.select('features', 'is_delayed', 'prediction').show(10)

# ---------------------------
# 5. Predict Single Record
# ---------------------------

print("Predicting a single record...")

# Example Record for Prediction
sample_data = [("Central", "Baker Street")]  # Example line and route
sample_df = spark.createDataFrame(sample_data, ['scala_tfl_underground.line', 'scala_tfl_underground.route'])

# Process the sample record
sample_df = line_indexer.fit(hive_df).transform(sample_df)
sample_df = route_indexer.fit(hive_df).transform(sample_df)
sample_df = assembler.transform(sample_df)

# Make Prediction
sample_prediction = model.transform(sample_df)
result = sample_prediction.select("prediction").collect()[0][0]

if result == 1.0:
    print("This journey is predicted to have a DELAY.")
else:
    print("This journey is predicted to be ON TIME.")

# ---------------------------
# 6. Save Predictions to Hive
# ---------------------------

result_df = predictions.drop('features', 'rawPrediction', 'probability')                        .withColumnRenamed('prediction', 'is_delayed_predicted')

result_df.write.mode("overwrite").saveAsTable("big_datajan2025.tfl_underground_predicted")

print("Successfully written to Hive table: big_datajan2025.tfl_underground_predicted")

# ---------------------------
# 7. Stop Spark Session
# ---------------------------

spark.stop()
