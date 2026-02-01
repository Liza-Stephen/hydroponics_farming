"""Main pipeline: Bronze -> Silver -> Gold -> Snowflake"""
from data_processing.bronze.batch_ingestion import run_bronze_ingestion
from data_processing.silver.processing import run_silver_processing
from data_processing.gold.processing import run_gold_processing
from data_processing.snowflake.processing import run_snowflake_processing
import os


def run_full_pipeline(source_csv_path=None):
    """Run complete pipeline"""
    print("="*60)
    print("HYDROPONICS DATA PROCESSING PIPELINE")
    print("="*60)
    
    print("\nBronze Layer...")
    run_bronze_ingestion(source_csv_path)
    
    print("\nSilver Layer...")
    run_silver_processing()
    
    print("\nGold Layer...")
    run_gold_processing()
    
    print("\nSnowflake Layer...")
    run_snowflake_processing()
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    run_full_pipeline(os.getenv("SOURCE_DATA_PATH"))
