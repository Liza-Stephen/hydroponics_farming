{{
  config(
    materialized='view',
    description='Intermediate model calculating sensor metrics and aggregations'
  )
}}

with sensor_data as (
    select * from {{ ref('stg_iot_data') }}
),

hourly_metrics as (
    select
        reading_date,
        reading_hour,
        
        -- Average sensor readings
        avg(ph_level) as avg_ph,
        avg(tds_level) as avg_tds,
        avg(air_temperature) as avg_air_temp,
        avg(air_humidity) as avg_air_humidity,
        avg(water_temperature) as avg_water_temp,
        avg(water_level) as avg_water_level,
        
        -- Min/Max values
        min(ph_level) as min_ph,
        max(ph_level) as max_ph,
        min(tds_level) as min_tds,
        max(tds_level) as max_tds,
        min(air_temperature) as min_air_temp,
        max(air_temperature) as max_air_temp,
        
        -- Standard deviations
        stddev(ph_level) as stddev_ph,
        stddev(tds_level) as stddev_tds,
        stddev(air_temperature) as stddev_air_temp,
        
        -- Counts
        count(*) as reading_count,
        sum(case when is_ph_optimal then 1 else 0 end) as ph_optimal_count,
        sum(case when is_tds_optimal then 1 else 0 end) as tds_optimal_count,
        sum(case when is_temp_optimal then 1 else 0 end) as temp_optimal_count,
        sum(case when is_humidity_optimal then 1 else 0 end) as humidity_optimal_count,
        sum(case when is_environment_optimal then 1 else 0 end) as environment_optimal_count,
        
        -- Equipment usage
        sum(case when ph_reducer_on then 1 else 0 end) as ph_reducer_activations,
        sum(case when add_water_on then 1 else 0 end) as water_additions,
        sum(case when nutrients_adder_on then 1 else 0 end) as nutrient_additions,
        sum(case when humidifier_on then 1 else 0 end) as humidifier_activations,
        sum(case when ex_fan_on then 1 else 0 end) as fan_activations
        
    from sensor_data
    group by reading_date, reading_hour
),

final as (
    select
        *,
        -- Calculate optimality percentages
        round((ph_optimal_count::float / reading_count) * 100, 2) as ph_optimality_pct,
        round((tds_optimal_count::float / reading_count) * 100, 2) as tds_optimality_pct,
        round((temp_optimal_count::float / reading_count) * 100, 2) as temp_optimality_pct,
        round((humidity_optimal_count::float / reading_count) * 100, 2) as humidity_optimality_pct,
        round((environment_optimal_count::float / reading_count) * 100, 2) as environment_optimality_pct
        
    from hourly_metrics
)

select * from final
