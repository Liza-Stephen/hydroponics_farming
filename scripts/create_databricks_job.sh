#!/bin/bash
# Script to create a Databricks job from local
# Requires: databricks CLI configured (databricks configure)

echo "Creating Databricks job..."

# Create the job using the JSON configuration
databricks jobs create --json-file databricks_job_config.json

echo "Job created successfully!"
echo "You can now run the job using: databricks jobs run-now --job-id <job-id>"
