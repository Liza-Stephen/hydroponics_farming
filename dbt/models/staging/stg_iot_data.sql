{{
  config(
    materialized='view',
    description='Staging model for IoT sensor data - cleans and prepares raw data'
  )
}}

with source as (
    select * from {{ var('source_database') }}.{{ var('source_schema') }}.iot_data
),

renamed as (
    select
        -- Primary keys
        reading_id,
        timestamp_key,
        
        -- Sensor readings
        ph_level,
        tds_level,
        water_level,
        air_temperature,
        air_humidity,
        water_temperature,
        
        -- Equipment states (boolean) - convert VARCHAR to BOOLEAN
        try_to_boolean(is_ph_reducer_on) as ph_reducer_on,
        try_to_boolean(is_add_water_on) as add_water_on,
        try_to_boolean(is_nutrients_adder_on) as nutrients_adder_on,
        try_to_boolean(is_humidifier_on) as humidifier_on,
        try_to_boolean(is_ex_fan_on) as ex_fan_on,
        
        -- Optimal condition indicators - convert VARCHAR to BOOLEAN
        try_to_boolean(is_ph_optimal) as is_ph_optimal,
        try_to_boolean(is_tds_optimal) as is_tds_optimal,
        try_to_boolean(is_temp_optimal) as is_temp_optimal,
        try_to_boolean(is_humidity_optimal) as is_humidity_optimal,
        
        -- Metadata
        ingestion_timestamp,
        source_file
        
    from source
),

final as (
    select
        *,
        -- Add calculated fields
        case 
            when is_ph_optimal = true 
                and is_tds_optimal = true 
                and is_temp_optimal = true 
                and is_humidity_optimal = true 
            then true 
            else false 
        end as is_environment_optimal,
        
        -- Calculate time-based fields
        date(timestamp_key) as reading_date,
        hour(timestamp_key) as reading_hour
        
    from renamed
)

select * from final
