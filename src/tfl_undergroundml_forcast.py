from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------
spark = SparkSession.builder.appName("TFL_Underground_Status_Forecast").enableHiveSupport().getOrCreate()

# ---------------------------
# 2. Load Data from Hive
# ---------------------------
hive_df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
hive_df.show(5)

# Data Cleaning: Handle missing values
hive_df = hive_df.fillna({
    'line': 'Unknown',
    'route': 'Unknown',
    'status': 'Unknown'  # Handle missing status if needed
})

# Ensure delay_time is numeric, but we won't use it in the features
hive_df = hive_df.withColumn("delay_time", col("delay_time").cast("int"))

# Identify correct timestamp column
timestamp_col = [col_name for col_name in hive_df.columns if 'timestamp' in col_name.lower()][0]

# Extract year and month for forecasting
hive_df = hive_df.withColumn("year", year(col(timestamp_col))).withColumn("month", month(col(timestamp_col)))

# ---------------------------
# 3. Feature Engineering (without delay_time)
# ---------------------------

# Index categorical columns (line, route, and status)
line_indexer = StringIndexer(inputCol='line', outputCol='line_index', handleInvalid='skip')
route_indexer = StringIndexer(inputCol='route', outputCol='route_index', handleInvalid='skip')
status_indexer = StringIndexer(inputCol='status', outputCol='status_index', handleInvalid='skip')

# Assemble features (excluding delay_time)
assembler = VectorAssembler(inputCols=['line_index', 'route_index', 'year', 'month'], outputCol='features')

# ---------------------------
# 4. Data Preprocessing and Check for Nulls
# ---------------------------

# Apply StringIndexers and assemble features
prepared_df = line_indexer.fit(hive_df).transform(hive_df)
prepared_df = route_indexer.fit(prepared_df).transform(prepared_df)
prepared_df = status_indexer.fit(prepared_df).transform(prepared_df)

# Ensure no null values in important columns
prepared_df = prepared_df.fillna({'line_index': 0, 'route_index': 0, 'status_index': 0})

# Assemble features and check for nulls in the feature column
prepared_df = assembler.transform(prepared_df)

# Check for null values in features and status_index
prepared_df = prepared_df.dropna(subset=["line_index", "route_index", "status_index", "features"])

# ---------------------------
# 5. Build and Train Forecast Model
# ---------------------------

print("Training the forecasting model...")

# Split data into training and testing sets
train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
print("Training Set: {} rows, Testing Set: {} rows".format(train_df.count(), test_df.count()))

# Linear Regression Model for Forecasting (status as label)
lr_classifier = LinearRegression(featuresCol='features', labelCol='status_index', maxIter=10)

# Train the model
status_model = lr_classifier.fit(train_df)

# Evaluate Model
predictions = status_model.transform(test_df)
predictions.select('features', 'status_index', 'prediction').show(10)

# ---------------------------
# 6. Forecast Future Status
# ---------------------------

print("Generating future status forecasts...")

# Prepare future data to forecast the status for the next few months
future_data = []
lines = hive_df.select("line").distinct().rdd.flatMap(lambda x: x).collect()

for line_name in lines:
    print("Forecasting for line: {}".format(line_name))
    for year_val in [2025]:
        for month_val in range(4, 7):  # Forecast for April, May, June
            future_data.append((line_name, "Unknown", year_val, month_val))

# Create a DataFrame for future predictions
future_df = spark.createDataFrame(future_data, ['line', 'route', 'year', 'month'])

# Apply indexers for future data
future_df = line_indexer.transform(future_df)
future_df = route_indexer.transform(future_df)

# Assemble features for future data (exclude status_index as we are predicting it)
future_df = assembler.transform(future_df)

# Make predictions on future data
future_predictions = status_model.transform(future_df)
future_predictions.select("line", "year", "month", "prediction").show(20)

# ---------------------------
# 7. Save Forecast to Hive
# ---------------------------

result_df = future_predictions.select("line", "year", "month", "prediction")
result_df.write.mode("overwrite").saveAsTable("big_datajan2025.tfl_underground_forecast_status")

print("Successfully written to Hive table: big_datajan2025.tfl_underground_forecast_status")

# ---------------------------
# 8. Stop Spark Session
# ---------------------------

spark.stop()
