{{
  config(
    materialized='table',
    description='Daily aggregated metrics for environment monitoring and reporting'
  )
}}

with hourly_metrics as (
    select * from {{ ref('int_sensor_metrics') }}
),

daily_aggregates as (
    select
        reading_date,
        
        -- Daily averages
        avg(avg_ph) as daily_avg_ph,
        avg(avg_tds) as daily_avg_tds,
        avg(avg_air_temp) as daily_avg_air_temp,
        avg(avg_air_humidity) as daily_avg_air_humidity,
        avg(avg_water_temp) as daily_avg_water_temp,
        avg(avg_water_level) as daily_avg_water_level,
        
        -- Daily ranges
        min(min_ph) as daily_min_ph,
        max(max_ph) as daily_max_ph,
        min(min_tds) as daily_min_tds,
        max(max_tds) as daily_max_tds,
        min(min_air_temp) as daily_min_air_temp,
        max(max_air_temp) as daily_max_air_temp,
        
        -- Daily totals
        sum(reading_count) as total_readings,
        sum(ph_optimal_count) as total_ph_optimal,
        sum(tds_optimal_count) as total_tds_optimal,
        sum(temp_optimal_count) as total_temp_optimal,
        sum(humidity_optimal_count) as total_humidity_optimal,
        sum(environment_optimal_count) as total_environment_optimal,
        
        -- Equipment usage totals
        sum(ph_reducer_activations) as total_ph_reducer_activations,
        sum(water_additions) as total_water_additions,
        sum(nutrient_additions) as total_nutrient_additions,
        sum(humidifier_activations) as total_humidifier_activations,
        sum(fan_activations) as total_fan_activations,
        
        -- Average optimality percentages
        avg(ph_optimality_pct) as avg_ph_optimality_pct,
        avg(tds_optimality_pct) as avg_tds_optimality_pct,
        avg(temp_optimality_pct) as avg_temp_optimality_pct,
        avg(humidity_optimality_pct) as avg_humidity_optimality_pct,
        avg(environment_optimality_pct) as avg_environment_optimality_pct
        
    from hourly_metrics
    group by reading_date
),

final as (
    select
        *,
        -- Calculate daily health score
        round(
            (avg_ph_optimality_pct + avg_tds_optimality_pct + 
             avg_temp_optimality_pct + avg_humidity_optimality_pct) / 4,
            2
        ) as daily_health_score,
        
        -- Calculate equipment efficiency
        round(
            (total_environment_optimal::float / total_readings) * 100,
            2
        ) as daily_environment_efficiency_pct
        
    from daily_aggregates
)

select * from final
order by reading_date desc
