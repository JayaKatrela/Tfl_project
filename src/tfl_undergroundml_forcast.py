""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, date_format
from pyspark.ml.feature import StringIndexer
from fbprophet import Prophet
import pandas as pd

# -------------------------------------
# 1. Initialize Spark Session
# -------------------------------------
spark = SparkSession.builder.appName("TFL_Underground_Delay_Forecasting").enableHiveSupport().getOrCreate()

# -------------------------------------
# 2. Load Data from Hive
# -------------------------------------
hive_df = spark.sql("SELECT * FROM big_datajan2025.scala_tfl_underground")
hive_df.show(5)

# -------------------------------------
# 3. Data Preparation
# -------------------------------------

# Handle missing values
hive_df = hive_df.fillna({'line': 'Unknown', 'route': 'Unknown', 'delay_time': '0', 'status': 'Good Service'})

# Convert timestamp to date and extract useful time-based features
hive_df = hive_df.withColumn("date", to_date(col("timestamp")))
hive_df = hive_df.withColumn("day_of_week", date_format(col("date"), "EEEE"))

# Create binary label: 1 if delayed, 0 if on-time
hive_df = hive_df.withColumn("is_delayed", when(col("status").contains("Delay"), 1).otherwise(0))

# -------------------------------------
# 4. Forecast Delays for Each Line
# -------------------------------------

def forecast_for_line(line_df, line_name, periods=30):
    print(f"Forecasting for line: {line_name}")
    
    # Convert Spark DataFrame to Pandas for Prophet
    pd_df = line_df.select("date", "is_delayed").groupBy("date").sum("is_delayed").toPandas()
    pd_df.columns = ["ds", "y"]  # Prophet requires these column names

    # Initialize and train the Prophet model
    model = Prophet()
    model.fit(pd_df)

    # Create future dates for prediction
    future = model.make_future_dataframe(periods=periods)

    # Forecast
    forecast = model.predict(future)
    forecast["line"] = line_name
    
    return forecast

# Get unique lines for forecasting
lines = [row.line for row in hive_df.select("line").distinct().collect()]

forecast_results = []

# Forecast delays for each line
for line in lines:
    line_df = hive_df.filter(col("line") == line)
    forecast = forecast_for_line(line_df, line, periods=30)  # Forecast for 30 days
    forecast_results.append(forecast)

# Combine all forecasts
forecast_df = pd.concat(forecast_results)

# -------------------------------------
# 5. Save Forecasts to Hive
# -------------------------------------

# Convert Pandas DataFrame back to Spark DataFrame
spark_forecast_df = spark.createDataFrame(forecast_df)

# Save predictions to Hive table
spark_forecast_df.write.mode("overwrite").saveAsTable("big_datajan2025.tfl_underground_forecast")

print("Forecast successfully saved to Hive: big_datajan2025.tfl_underground_forecast")


# -------------------------------------
# 6. Stop Spark Session
# -------------------------------------

spark.stop()