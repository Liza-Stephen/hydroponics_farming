{{
  config(
    materialized='table',
    description='Fact table for sensor readings with business-friendly columns and metrics'
  )
}}

with sensor_data as (
    select * from {{ ref('stg_iot_data') }}
),

time_dim as (
    select * from {{ ref('stg_time_dimension') }}
),

final as (
    select
        -- Fact table primary key
        s.reading_id,
        s.timestamp_key,
        
        -- Time dimension attributes
        t.date as reading_date,
        t.hour as reading_hour,
        t.day_name,
        t.month_name,
        t.day_type,  -- Weekend/Weekday
        
        -- Sensor readings (business-friendly names)
        s.ph_level as ph_value,
        s.tds_level as tds_value,
        s.air_temperature as air_temp_celsius,
        s.air_humidity as air_humidity_pct,
        s.water_temperature as water_temp_celsius,
        s.water_level as water_level_pct,
        
        -- Equipment status
        s.ph_reducer_on,
        s.add_water_on,
        s.nutrients_adder_on,
        s.humidifier_on,
        s.ex_fan_on,
        
        -- Optimal condition flags
        s.is_ph_optimal,
        s.is_tds_optimal,
        s.is_temp_optimal,
        s.is_humidity_optimal,
        s.is_environment_optimal,
        
        -- Calculated metrics
        case 
            when s.ph_level < 5.5 then 'Too Acidic'
            when s.ph_level > 6.5 then 'Too Alkaline'
            else 'Optimal'
        end as ph_status,
        
        case 
            when s.tds_level < 800 then 'Low Nutrients'
            when s.tds_level > 1200 then 'High Nutrients'
            else 'Optimal'
        end as tds_status,
        
        case 
            when s.air_temperature < 20 then 'Too Cold'
            when s.air_temperature > 30 then 'Too Hot'
            else 'Optimal'
        end as temperature_status,
        
        -- Metadata
        s.ingestion_timestamp,
        s.source_file
        
    from sensor_data s
    left join time_dim t
        on s.timestamp_key = t.timestamp_key
)

select * from final
