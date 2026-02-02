{{
  config(
    materialized='view',
    description='Staging model for time dimension table'
  )
}}

select * from {{ var('source_database') }}.{{ var('source_schema') }}.dim_time
