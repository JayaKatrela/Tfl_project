from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, month, year, current_date, date_add
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------
spark = SparkSession.builder.appName("TFL_Underground_Status_Forecast").enableHiveSupport().getOrCreate()

# ---------------------------
# 2. Load Data from Hive
# ---------------------------
hive_df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
hive_df.show(10)

# Data Cleaning: Handle missing values
hive_df = hive_df.fillna({
    'line': 'Unknown',
    'route': 'Unknown',
    'status': 'Unknown'
})

# Ensure 'status' is categorical (you can use StringIndexer later)
hive_df = hive_df.withColumn("status", col("status").cast("string"))

# Identify correct timestamp column
timestamp_col = [col_name for col_name in hive_df.columns if 'timestamp' in col_name.lower()][0]

# Extract year and month for forecasting
hive_df = hive_df.withColumn("year", year(col(timestamp_col))).withColumn("month", month(col(timestamp_col)))

# ---------------------------
# 3. Feature Engineering
# ---------------------------
# Index categorical columns
line_indexer = StringIndexer(inputCol='line', outputCol='line_index', handleInvalid='skip')
route_indexer = StringIndexer(inputCol='route', outputCol='route_index', handleInvalid='skip')

# Assemble features
assembler = VectorAssembler(inputCols=['line_index', 'route_index', 'year', 'month'], outputCol='features')

# ---------------------------
# 4. Data Preprocessing and Check for Nulls
# ---------------------------

# Apply StringIndexer and check for null values in critical columns
prepared_df = line_indexer.fit(hive_df).transform(hive_df)
prepared_df = route_indexer.fit(prepared_df).transform(prepared_df)

# Ensure no null values in important columns
prepared_df = prepared_df.fillna({'line_index': 0, 'route_index': 0, 'status': 'Unknown'})

# Assemble features and check for nulls in the feature column
prepared_df = assembler.transform(prepared_df)

# Drop any rows with null values in the important columns
prepared_df = prepared_df.dropna(subset=["line_index", "route_index", "features", "status"])

# ---------------------------
# 5. Build and Train Classification Model
# ---------------------------
print("Training the classification model for 'status'...")

# Split data into training and testing sets
train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
print("Training Set: {} rows, Testing Set: {} rows".format(train_df.count(), test_df.count()))

# Logistic Regression Model for Forecasting (using status as label)
lr_classifier = LogisticRegression(featuresCol='features', labelCol='status_index', maxIter=10)

# Index the 'status' column for classification
status_indexer = StringIndexer(inputCol='status', outputCol='status_index')
prepared_df = status_indexer.fit(prepared_df).transform(prepared_df)

# Train the model
status_model = lr_classifier.fit(train_df)

# Evaluate Model
predictions = status_model.transform(test_df)
predictions.select('features', 'status', 'prediction').show(10)

# ---------------------------
# 6. Forecast for the Coming Week
# ---------------------------

print("Generating future forecasts for the coming week...")

# Generate future data for next week (assuming you forecast for specific lines and dates)
# Assuming we're forecasting for the coming week: get the current date
today = current_date()

# Generate the future dates (coming week)
future_data = []
lines = hive_df.select("line").distinct().rdd.flatMap(lambda x: x).collect()

# Forecast for the next week (you can adjust the dates accordingly)
for line_name in lines:
    print("Forecasting for line: {}".format(line_name))
    for days_to_add in range(7):  # For each day in the upcoming week
        future_date = date_add(today, days_to_add)  # Add days to current date
        future_year = future_date.year
        future_month = future_date.month
        future_data.append((line_name, "Unknown", future_year, future_month))

# Create a DataFrame for future predictions
future_df = spark.createDataFrame(future_data, ['line', 'route', 'year', 'month'])

# Index future data and assemble features
future_df = line_indexer.fit(hive_df).transform(future_df)
future_df = route_indexer.fit(hive_df).transform(future_df)
future_df = assembler.transform(future_df)

# Make predictions on future data
future_predictions = status_model.transform(future_df)
future_predictions.select("line", "year", "month", "prediction").show(20)

# ---------------------------
# 7. Save Forecast to Hive
# ---------------------------
result_df = future_predictions.select("line", "year", "month", "prediction")
result_df.write.mode("overwrite").saveAsTable("big_datajan2025.tfl_underground_forecast")

print("Successfully written forecast to Hive table: big_datajan2025.tfl_underground_forecast")

# ---------------------------
# 8. Stop Spark Session
# ---------------------------
spark.stop()
