{{
  config(
    materialized='table',
    description='Dimension table for sensor health status and classifications'
  )
}}

with sensor_data as (
    select * from {{ ref('stg_iot_data') }}
),

health_classifications as (
    select distinct
        reading_id,
        timestamp_key,
        
        -- Health score (0-100)
        case
            when is_environment_optimal then 100
            when is_ph_optimal and is_tds_optimal and is_temp_optimal then 75
            when is_ph_optimal and is_tds_optimal then 50
            when is_ph_optimal or is_tds_optimal then 25
            else 0
        end as health_score,
        
        -- Health status category
        case
            when is_environment_optimal then 'Excellent'
            when is_ph_optimal and is_tds_optimal and is_temp_optimal then 'Good'
            when is_ph_optimal and is_tds_optimal then 'Fair'
            when is_ph_optimal or is_tds_optimal then 'Poor'
            else 'Critical'
        end as health_status,
        
        -- Alert flags
        case when not is_ph_optimal then 1 else 0 end as ph_alert,
        case when not is_tds_optimal then 1 else 0 end as tds_alert,
        case when not is_temp_optimal then 1 else 0 end as temp_alert,
        case when not is_humidity_optimal then 1 else 0 end as humidity_alert,
        
        -- Action required flags
        case 
            when ph_level < 5.0 or ph_level > 7.0 then 1
            else 0
        end as ph_action_required,
        
        case 
            when tds_level < 600 or tds_level > 1400 then 1
            else 0
        end as tds_action_required
        
    from sensor_data
)

select * from health_classifications
