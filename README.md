# Hydroponics Farming Data Processing Pipeline

A Databricks-based data processing pipeline implementing the medallion architecture (Bronze → Silver → Gold) for IoT sensor data from hydroponics farming systems.

## Execution Mode

**Code runs natively in Databricks via Repos and Jobs.**

- Code is stored in GitHub and synced to Databricks via Repos
- Data is uploaded to DBFS before running
- Pipeline runs as a Databricks Job (triggered from local via CLI)
- All execution happens natively in Databricks (no Databricks Connect needed)
- Supports serverless compute or traditional clusters via job configuration

## Architecture Overview

This project implements a three-layer medallion architecture:

- **Bronze Layer**: Raw data ingestion from CSV files in DBFS
- **Silver Layer**: Cleaned, validated, and deduplicated data
- **Gold Layer**: Business-ready fact and dimension tables in star schema format

## Project Structure

```
hydroponics_farming/
├── config/
│   ├── databricks_config.py      # Databricks configuration
│   └── config_template.env        # Environment variables template
├── src/
│   ├── data_processing/
│   │   ├── bronze_ingestion.py   # Bronze layer ingestion
│   │   ├── silver_processing.py  # Silver layer cleaning & validation
│   │   └── gold_processing.py    # Gold layer fact & dimension tables
│   └── main.py                   # Main orchestration script
├── scripts/
│   ├── upload_data_to_dbfs.py    # Script to upload data to DBFS
│   └── create_databricks_job.sh   # Script to create Databricks job
├── raw_data/
│   └── iot_data_raw.csv          # Source IoT sensor data
├── requirements.txt              # Python dependencies
├── databricks_job_config.json    # Databricks job configuration
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Python 3.8+ (for local scripts)
- Databricks workspace access
- GitHub repository
- Databricks CLI (`brew install databricks/tap/databricks`)

### Setup Steps

#### 1. Push Code to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/hydroponics_farming.git
git push -u origin main
```

#### 2. Connect Repo to Databricks

1. In Databricks UI → **Repos** → **Add Repo**
2. Select **GitHub**, enter repo URL, select branch `main`
3. Code available at: `/Repos/your-username/hydroponics_farming`

#### 3. Configure Databricks CLI

```bash
brew install databricks/tap/databricks
databricks configure
# Enter workspace URL and personal access token
```

#### 4. Upload Data to DBFS

```bash
python scripts/upload_data_to_dbfs.py
```

Verify upload:
```bash
databricks fs ls dbfs:/mnt/hydroponics/raw_data/
```

#### 5. Create Databricks Job

**Option A: Using script**
```bash
# Edit databricks_job_config.json:
# - Set DATABRICKS_CATALOG (e.g., "main")
# - Update path: /Repos/your-username/hydroponics_farming/src/main.py
bash scripts/create_databricks_job.sh
```

**Option B: Using UI**
1. **Workflows** → **Jobs** → **Create Job**
2. **Task type**: Python script
3. **Path**: `/Repos/your-username/hydroponics_farming/src/main.py`
4. **Environment variables**:
   - `DATABRICKS_CATALOG=your_catalog`
   - `SOURCE_DATA_PATH=dbfs:/mnt/hydroponics/raw_data/iot_data_raw.csv`

#### 6. Run the Job

**CLI:**
```bash
databricks jobs run-now --job-id <job-id>
```

**UI:** Workflows → Jobs → Run Now

#### 7. Verify Results

```sql
USE CATALOG your_catalog;
SELECT COUNT(*) FROM bronze.iot_data;
SELECT COUNT(*) FROM silver.iot_data;
SELECT COUNT(*) FROM gold.iot_data;
SELECT COUNT(*) FROM gold.dim_time;
SELECT COUNT(*) FROM gold.dim_equipment;
```

## Configuration

### Environment Variables (Set in Job Configuration)

- **DATABRICKS_CATALOG**: Unity Catalog catalog name (e.g., `main`, `production`)
- **SOURCE_DATA_PATH**: Path to source CSV in DBFS (default: `dbfs:/mnt/hydroponics/raw_data/iot_data_raw.csv`)

### Data Paths

All data is stored in DBFS:
- Bronze: `dbfs:/mnt/hydroponics/bronze/iot_data`
- Silver: `dbfs:/mnt/hydroponics/silver/iot_data`
- Gold: `dbfs:/mnt/hydroponics/gold/iot_data`

Tables are registered in Unity Catalog:
- `{catalog}.bronze.iot_data`
- `{catalog}.silver.iot_data`
- `{catalog}.gold.iot_data`
- `{catalog}.gold.dim_time`
- `{catalog}.gold.dim_equipment`

## Usage

### Run Full Pipeline

The pipeline runs automatically when the Databricks job is executed. It processes:
1. **Bronze Layer**: Ingests raw CSV data from DBFS
2. **Silver Layer**: Cleans and validates data
3. **Gold Layer**: Creates fact and dimension tables

### Run Individual Layers

To run individual layers in Databricks notebooks:

```python
# Bronze only
from data_processing.bronze_ingestion import run_bronze_ingestion
run_bronze_ingestion("dbfs:/mnt/hydroponics/raw_data/iot_data_raw.csv")

# Silver only (requires bronze to exist)
from data_processing.silver_processing import run_silver_processing
run_silver_processing()

# Gold only (requires silver to exist)
from data_processing.gold_processing import run_gold_processing
run_gold_processing()
```

## Data Processing Details

### Bronze Layer
- Ingests raw CSV data from DBFS
- Preserves all original data
- Adds metadata: `ingestion_timestamp`, `source_file`
- Stores in Delta format for ACID transactions

### Silver Layer
- Type conversions (string to numeric, ON/OFF to boolean)
- Data quality validation:
  - pH range: 0-14
  - TDS: non-negative
  - Temperature: -10°C to 50°C
  - Humidity: 0-100%
- Deduplication based on id and timestamp
- Adds `silver_processed_timestamp`

### Gold Layer
- Creates star schema with fact and dimension tables
- Calculates optimal condition indicators
- Enables efficient analytical queries
- Time dimension for temporal analysis
- Equipment dimension for equipment tracking

## Querying the Data

After processing, you can query the tables in Databricks:

```sql
-- Example: Find readings with suboptimal pH
SELECT 
    t.date,
    t.hour,
    f.ph_level,
    f.is_ph_optimal,
    f.air_temperature
FROM your_catalog.gold.iot_data f
JOIN your_catalog.gold.dim_time t
    ON f.timestamp_key = t.timestamp_key
WHERE f.is_ph_optimal = false
ORDER BY t.date DESC, t.hour DESC;
```

## Troubleshooting

**Job fails: "Catalog not found"**
- Verify `DATABRICKS_CATALOG` in job environment variables
- Ensure you have permissions to access the catalog

**Job fails: "File not found"**
- Check data uploaded: `databricks fs ls dbfs:/mnt/hydroponics/raw_data/`
- Verify `SOURCE_DATA_PATH` in job config matches the uploaded path

**Job fails: "Module not found"**
- Ensure the Repo path in job configuration is correct
- Verify all Python files are in the Repo

**Schema creation errors**
- Check that you have CREATE SCHEMA permissions in the catalog
- Verify the catalog name is correct

**Data not found**
- Verify data is uploaded to DBFS using `scripts/upload_data_to_dbfs.py`

**Schema errors**
- Check that the source CSV matches the expected format

**Permission errors**
- Verify your job has write access to the specified catalog/schema
