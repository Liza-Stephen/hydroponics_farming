"""Gold Layer: Fact and dimension tables"""
from pyspark.sql.functions import col, year, month, dayofmonth, hour, minute, date_format, dayofweek, weekofyear, quarter, when
from config.databricks_config import get_spark_session


def create_time_dimension(spark, silver_table_name, time_table_name):
    """Create time dimension"""
    df_silver = spark.table(silver_table_name)
    df_time = df_silver.select(
        col("timestamp").alias("timestamp_key"),
        date_format(col("timestamp"), "yyyy-MM-dd HH:mm:ss").alias("timestamp_full"),
        date_format(col("timestamp"), "yyyy-MM-dd").alias("date"),
        date_format(col("timestamp"), "HH:mm:ss").alias("time"),
        year(col("timestamp")).alias("year"),
        quarter(col("timestamp")).alias("quarter"),
        month(col("timestamp")).alias("month"),
        weekofyear(col("timestamp")).alias("week_of_year"),
        dayofmonth(col("timestamp")).alias("day_of_month"),
        dayofweek(col("timestamp")).alias("day_of_week"),
        hour(col("timestamp")).alias("hour"),
        minute(col("timestamp")).alias("minute"),
        date_format(col("timestamp"), "EEEE").alias("day_name"),
        date_format(col("timestamp"), "MMMM").alias("month_name"),
        when(dayofweek(col("timestamp")).isin([1, 7]), "Weekend")
            .otherwise("Weekday").alias("day_type")
    ).distinct()
    
    df_time.write.format("delta").mode("overwrite").saveAsTable(time_table_name)
    return time_table_name


def create_equipment_dimension(spark, equipment_table_name):
    """Create equipment dimension"""
    equipment_data = [
        (1, "pH_reducer", "pH Reduction System", "Controls pH levels in nutrient solution"),
        (2, "add_water", "Water Addition System", "Adds water to maintain level"),
        (3, "nutrients_adder", "Nutrient Addition System", "Adds nutrients to solution"),
        (4, "humidifier", "Humidifier", "Controls humidity in growing environment"),
        (5, "ex_fan", "Exhaust Fan", "Ventilation system for air circulation")
    ]
    
    df_equipment = spark.createDataFrame(
        equipment_data,
        ["equipment_id", "equipment_code", "equipment_name", "equipment_description"]
    )
    df_equipment.write.format("delta").mode("overwrite").saveAsTable(equipment_table_name)
    return equipment_table_name


def create_fact_sensor_readings(spark, silver_table_name, fact_table_name):
    """Create fact table"""
    df_silver = spark.table(silver_table_name)
    df_fact = df_silver.select(
        col("id").alias("reading_id"),
        col("timestamp").alias("timestamp_key"),
        col("pH").alias("ph_level"),
        col("TDS").alias("tds_level"),
        col("water_level"),
        col("DHT_temp").alias("air_temperature"),
        col("DHT_humidity").alias("air_humidity"),
        col("water_temp").alias("water_temperature"),
        col("pH_reducer").alias("is_ph_reducer_on"),
        col("add_water").alias("is_add_water_on"),
        col("nutrients_adder").alias("is_nutrients_adder_on"),
        col("humidifier").alias("is_humidifier_on"),
        col("ex_fan").alias("is_ex_fan_on"),
        when(col("pH").between(5.5, 6.5), True).otherwise(False).alias("is_ph_optimal"),
        when(col("TDS").between(800, 1200), True).otherwise(False).alias("is_tds_optimal"),
        when(col("DHT_temp").between(20, 28), True).otherwise(False).alias("is_temp_optimal"),
        when(col("DHT_humidity").between(40, 70), True).otherwise(False).alias("is_humidity_optimal"),
        col("ingestion_timestamp"),
        col("source_file")
    )
    df_fact.write.format("delta").mode("overwrite").saveAsTable(fact_table_name)
    return fact_table_name


def create_gold_tables(spark, config):
    """Create all gold layer tables"""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.gold_schema}")
    
    silver_table_name = config.get_table_name(config.silver_schema, "iot_data")
    time_table_name = config.get_table_name(config.gold_schema, "dim_time")
    equipment_table_name = config.get_table_name(config.gold_schema, "dim_equipment")
    fact_table_name = config.get_table_name(config.gold_schema, "iot_data")
    
    # Create managed tables first
    spark.sql(f"CREATE TABLE IF NOT EXISTS {time_table_name} USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {equipment_table_name} USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {fact_table_name} USING DELTA")
    
    # Create dimensions and fact
    create_time_dimension(spark, silver_table_name, time_table_name)
    create_equipment_dimension(spark, equipment_table_name)
    create_fact_sensor_readings(spark, silver_table_name, fact_table_name)
    
    return {"dim_time": time_table_name, "dim_equipment": equipment_table_name, "iot_data": fact_table_name}


def run_gold_processing():
    """Run gold layer processing"""
    spark, config = get_spark_session()
    tables = create_gold_tables(spark, config)
    
    fact_table = config.get_table_name(config.gold_schema, "iot_data")
    print("\nSample from fact table:")
    spark.sql(f"SELECT * FROM {fact_table} LIMIT 5").show(truncate=False)
    
    return tables


if __name__ == "__main__":
    run_gold_processing()
