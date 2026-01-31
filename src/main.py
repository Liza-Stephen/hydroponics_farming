"""Main pipeline: Bronze -> Silver -> Gold"""
from data_processing.bronze_ingestion import run_bronze_ingestion
from data_processing.silver_processing import run_silver_processing
from data_processing.gold_processing import run_gold_processing
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
    
    print("\n✓ Pipeline completed!")


if __name__ == "__main__":
    run_full_pipeline(os.getenv("SOURCE_DATA_PATH"))
