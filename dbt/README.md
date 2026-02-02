# Hydroponics Semantic Layer (dbt)

This dbt project creates a semantic layer on top of Snowflake data, providing business-friendly views, metrics, and KPIs for hydroponics farming analytics.

**Note:** This dbt project runs **exclusively in Databricks** as part of the data processing pipeline. It is executed automatically after Snowflake data loading completes.

## Overview

The semantic layer transforms raw Snowflake data into:
- **Business-friendly column names** (e.g., `ph_value` instead of `ph_level`)
- **Categorized status fields** (e.g., `ph_status`: "Too Acidic", "Optimal", "Too Alkaline")
- **Pre-calculated metrics** (e.g., health scores, optimality percentages)
- **Aggregated KPIs** (e.g., daily environment efficiency)

## Project Structure

```
dbt/
├── dbt_project.yml          # dbt project configuration
├── profiles.yml             # Snowflake connection profiles
├── schema.yml               # Model documentation and tests
├── models/
│   ├── staging/             # Staging models (clean source data)
│   │   ├── stg_iot_data.sql
│   │   └── stg_time_dimension.sql
│   ├── intermediate/        # Intermediate models (business logic)
│   │   └── int_sensor_metrics.sql
│   ├── marts/              # Final business-ready tables
│   │   ├── fct_sensor_readings.sql
│   │   └── dim_sensor_health.sql
│   └── metrics/            # KPI and metric tables
│       └── daily_environment_metrics.sql
└── README.md               # This file
```

## Setup

### 1. Configure Databricks Environment

The dbt semantic layer runs in Databricks as part of the data processing job. 

**Required Databricks Environment:**
- Create a Databricks environment with dependency: `dbt-snowflake>=1.6.0`
- Environment key: `dbt_env` (or update the job config to match your environment key)

### 2. Job Configuration

The dbt task is automatically included in the `data_processing_with_dbt.json` job configuration:

```json
{
  "task_key": "semantic_layer_dbt",
  "depends_on": [{"task_key": "snowflake_processing"}],
  "spark_python_task": {
    "python_file": "/Repos/.../src/data_processing/semantic_layer/dbt_runner.py",
    "parameters": [
      "SNOWFLAKE_ACCOUNT",
      "SNOWFLAKE_USER", 
      "SNOWFLAKE_PASSWORD",
      "SNOWFLAKE_WAREHOUSE",
      "SNOWFLAKE_DATABASE",
      "SNOWFLAKE_SCHEMA",
      "prod"  // dbt target (dev or prod)
    ]
  },
  "environment_key": "dbt_env"
}
```

### 3. Execution Flow

The semantic layer runs automatically as part of the pipeline:

```
Bronze → Silver → Gold → Snowflake → Semantic Layer (dbt)
```

When the job runs:
1. Data flows through Bronze, Silver, Gold layers
2. Snowflake processing loads data to Snowflake
3. **dbt automatically runs** to create semantic layer models
4. Business-friendly views are available in Snowflake

### 4. Manual Execution (for testing/debugging)

If you need to run dbt manually in Databricks for testing:

1. Create a Databricks notebook or Python script
2. Use the `dbt_runner.py` script or call dbt commands directly:

```python
import subprocess
import os

# Set environment variables
os.environ["SNOWFLAKE_ACCOUNT"] = "your-account"
os.environ["SNOWFLAKE_USER"] = "your-user"
# ... etc

# Run dbt
subprocess.run(["dbt", "run", "--target", "dev"])
```

## Models

### Staging Models

**`stg_iot_data`**: Cleans and prepares raw IoT sensor data
- Renames columns to business-friendly names
- Adds calculated fields (e.g., `is_environment_optimal`)
- Adds time-based fields (date, hour)

**`stg_time_dimension`**: Staging for time dimension table

### Intermediate Models

**`int_sensor_metrics`**: Hourly aggregated sensor metrics
- Average, min, max values per hour
- Optimality counts and percentages
- Equipment activation counts
- Standard deviations for variability analysis

### Mart Models (Business-Ready Tables)

**`fct_sensor_readings`**: Fact table with business-friendly columns
- Business-friendly column names
- Status categorizations (pH status, TDS status, temperature status)
- Joined with time dimension for temporal analysis

**`dim_sensor_health`**: Health scoring and alerting dimension
- Health score (0-100)
- Health status categories (Excellent, Good, Fair, Poor, Critical)
- Alert flags for each sensor type
- Action required flags

### Metrics Models

**`daily_environment_metrics`**: Daily aggregated KPIs
- Daily averages and ranges
- Total optimal readings
- Equipment usage totals
- Daily health scores
- Environment efficiency percentages

## Usage Examples

### Query Business-Friendly Data

```sql
-- Get sensor readings with status classifications
SELECT 
    reading_date,
    ph_value,
    ph_status,
    tds_value,
    tds_status,
    air_temp_celsius,
    temperature_status,
    is_environment_optimal
FROM marts.fct_sensor_readings
WHERE reading_date >= CURRENT_DATE - 7
ORDER BY timestamp_key DESC;
```

### Get Health Scores

```sql
-- Get health scores and alerts
SELECT 
    timestamp_key,
    health_score,
    health_status,
    ph_alert,
    tds_alert,
    temp_alert,
    ph_action_required
FROM marts.dim_sensor_health
WHERE health_status IN ('Poor', 'Critical')
ORDER BY timestamp_key DESC;
```

### Daily KPIs

```sql
-- Get daily environment metrics
SELECT 
    reading_date,
    daily_health_score,
    daily_environment_efficiency_pct,
    total_environment_optimal,
    total_readings,
    daily_avg_ph,
    daily_avg_tds
FROM metrics.daily_environment_metrics
ORDER BY reading_date DESC
LIMIT 30;
```

### Hourly Aggregations

```sql
-- Get hourly sensor metrics
SELECT 
    reading_date,
    reading_hour,
    avg_ph,
    ph_optimality_pct,
    environment_optimality_pct,
    ph_reducer_activations
FROM intermediate.int_sensor_metrics
WHERE reading_date = CURRENT_DATE
ORDER BY reading_hour;
```

## Data Quality Tests

dbt automatically runs data quality tests defined in `schema.yml`:

```bash
# Run all tests
dbt test

# Run tests for specific model
dbt test --select stg_iot_data
```

Tests include:
- Uniqueness checks
- Not null checks
- Accepted value ranges
- Referential integrity

## Documentation

Generate and view documentation:

```bash
dbt docs generate
dbt docs serve
```

This opens an interactive documentation site with:
- Model lineage graphs
- Column descriptions
- Test results
- Source table definitions

## Integration with Pipeline

The semantic layer is **fully integrated** into the Databricks job pipeline:

```
Bronze → Silver → Gold → Snowflake → Semantic Layer (dbt) ✨
```

**Execution:**
1. Data flows from Databricks Gold layer to Snowflake (via `snowflake/processing.py`)
2. **dbt automatically runs** after Snowflake processing completes
3. dbt models read from Snowflake source tables and create semantic views
4. Business-friendly views are immediately available for analytics tools

**Job Configuration:**
- Use `jobs/data_processing_with_dbt.json` for the complete pipeline
- The dbt task depends on `snowflake_processing` task
- Runs automatically on the same schedule as the main pipeline

## Next Steps

1. **Add more metrics**: Create additional metric models for specific KPIs
2. **Custom dimensions**: Add business-specific dimension tables
3. **Incremental models**: Convert large tables to incremental materialization
4. **Snapshots**: Track historical changes to key metrics
5. **Exposures**: Define how models are used in BI tools (Tableau, Power BI, etc.)
