# Import required libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session with Hive support
spark = SparkSession.builder \
    .appName("Hive to Spark ML") \
    .enableHiveSupport() \
    .getOrCreate()

# Load data from Hive table (replace 'database.table_name' with your actual Hive table)
hive_table = "database.table_name"
data = spark.sql(f"SELECT * FROM {hive_table}")

# Display the schema and a sample of the data
data.printSchema()
data.show(5)

# Handle categorical columns (example with label indexing)
label_indexer = StringIndexer(inputCol="target_column", outputCol="label")

# Assemble feature columns
feature_cols = [col for col in data.columns if col != 'target_column']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Define a simple Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create a pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, lr])

# Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Stop the Spark session
spark.stop()
