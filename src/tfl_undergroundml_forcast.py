from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, month, year
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------

spark = SparkSession.builder.appName("TFL_Underground_Delay_Forecast").enableHiveSupport().getOrCreate()

# ---------------------------
# 2. Load Data from Hive
# ---------------------------

hive_df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
hive_df.show(10)

# Data Cleaning: Handle missing values
hive_df = hive_df.fillna({
    'line': 'Unknown',
    'route': 'Unknown',
    'delay_time': '0'
})

# Ensure delay_time is numeric
hive_df = hive_df.withColumn("delay_time", col("delay_time").cast("int"))

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
# 4. Build and Train Forecast Model
# ---------------------------

print("Training the forecasting model...")

# Prepare data
prepared_df = line_indexer.fit(hive_df).transform(hive_df)
prepared_df = route_indexer.fit(prepared_df).transform(prepared_df)
prepared_df = assembler.transform(prepared_df)

# Split data into training and testing sets
train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
print("Training Set: {} rows, Testing Set: {} rows".format(train_df.count(), test_df.count()))

# Linear Regression Model for Forecasting
lr = LinearRegression(featuresCol='features', labelCol='delay_time', maxIter=10)

forecast_model = lr.fit(train_df)

# Evaluate Model
predictions = forecast_model.transform(test_df)
predictions.select('features', 'delay_time', 'prediction').show(10)

# ---------------------------
# 5. Forecast Future Delays
# ---------------------------

print("Generating future forecasts...")

future_data = []
lines = hive_df.select("line").distinct().rdd.flatMap(lambda x: x).collect()

for line_name in lines:
    print("Forecasting for line: {}".format(line_name))
    for year_val in [2025]:
        for month_val in range(4, 7):  # Forecast for April, May, June
            future_data.append((line_name, "Unknown", year_val, month_val))

future_df = spark.createDataFrame(future_data, ['line', 'route', 'year', 'month'])

future_df = line_indexer.fit(hive_df).transform(future_df)
future_df = route_indexer.fit(hive_df).transform(future_df)
future_df = assembler.transform(future_df)

future_predictions = forecast_model.transform(future_df)
future_predictions.select("line", "year", "month", "prediction").show(20)

# ---------------------------
# 6. Save Forecast to Hive
# ---------------------------

result_df = future_predictions.select("line", "year", "month", "prediction")
result_df.write.mode("overwrite").saveAsTable("big_datajan2025.tfl_underground_forecast")

print("Successfully written to Hive table: big_datajan2025.tfl_underground_forecast")

# ---------------------------
# 7. Stop Spark Session
# ---------------------------

spark.stop()
