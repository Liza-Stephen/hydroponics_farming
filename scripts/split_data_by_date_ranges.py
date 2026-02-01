#!/usr/bin/env python3
"""
Split CSV data into different ingestion patterns based on date ranges:
- Batch Backfill: 2023-11-26 to 2023-12-19 (single file)
- Incremental Batch: 2023-12-20 to 2023-12-23 (daily files)
- API/Streaming Replay: 2023-12-24 to 2023-12-26 (daily files)

Usage:
    python scripts/split_data_by_date_ranges.py --input raw_data/iot_data_raw.csv --output-dir data_splits/
"""

import argparse
import csv
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str.strip('"'), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def get_date_from_timestamp(timestamp_str):
    """Extract date (YYYY-MM-DD) from timestamp string."""
    dt = parse_timestamp(timestamp_str)
    if dt:
        return dt.date()
    return None


def split_data_by_date_ranges(input_file, output_dir):
    """
    Split CSV data into different ingestion patterns based on date ranges.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Base output directory
    """
    # Define date ranges
    batch_backfill_start = datetime(2023, 11, 26).date()
    batch_backfill_end = datetime(2023, 12, 19).date()
    
    incremental_start = datetime(2023, 12, 20).date()
    incremental_end = datetime(2023, 12, 23).date()
    
    api_replay_start = datetime(2023, 12, 24).date()
    api_replay_end = datetime(2023, 12, 26).date()
    
    # Create output directories
    batch_dir = os.path.join(output_dir, "batch_backfill")
    incremental_dir = os.path.join(output_dir, "incremental_batch")
    api_replay_dir = os.path.join(output_dir, "api_streaming_replay")
    
    Path(batch_dir).mkdir(parents=True, exist_ok=True)
    Path(incremental_dir).mkdir(parents=True, exist_ok=True)
    Path(api_replay_dir).mkdir(parents=True, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        
        # Find timestamp column index
        timestamp_idx = header.index('timestamp') if 'timestamp' in header else None
        if timestamp_idx is None:
            print("Error: 'timestamp' column not found in CSV")
            return
        
        # Organize rows by date and category
        batch_backfill_rows = []
        incremental_rows_by_date = defaultdict(list)
        api_replay_rows_by_date = defaultdict(list)
        other_rows = []
        
        total_rows = 0
        for row in reader:
            total_rows += 1
            if len(row) <= timestamp_idx:
                continue
            
            timestamp_str = row[timestamp_idx]
            date = get_date_from_timestamp(timestamp_str)
            
            if date is None:
                other_rows.append(row)
                continue
            
            # Categorize by date range
            if batch_backfill_start <= date <= batch_backfill_end:
                batch_backfill_rows.append(row)
            elif incremental_start <= date <= incremental_end:
                incremental_rows_by_date[date].append(row)
            elif api_replay_start <= date <= api_replay_end:
                api_replay_rows_by_date[date].append(row)
            else:
                other_rows.append(row)
    
    print(f"Total rows processed: {total_rows}")
    print("-" * 60)
    
    # Write Batch Backfill (single file)
    batch_file = os.path.join(batch_dir, "batch_backfill_2023-11-26_to_2023-12-19.csv")
    with open(batch_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(batch_backfill_rows)
    print(f"✓ Batch Backfill: {len(batch_backfill_rows)} rows → {batch_file}")
    
    # Write Incremental Batch (daily files)
    incremental_count = 0
    for date in sorted(incremental_rows_by_date.keys()):
        date_str = date.strftime("%Y-%m-%d")
        daily_file = os.path.join(incremental_dir, f"incremental_{date_str}.csv")
        with open(daily_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(incremental_rows_by_date[date])
        incremental_count += len(incremental_rows_by_date[date])
        print(f"✓ Incremental Batch {date_str}: {len(incremental_rows_by_date[date])} rows → {daily_file}")
    
    # Write API/Streaming Replay (daily files)
    api_replay_count = 0
    for date in sorted(api_replay_rows_by_date.keys()):
        date_str = date.strftime("%Y-%m-%d")
        daily_file = os.path.join(api_replay_dir, f"api_replay_{date_str}.csv")
        with open(daily_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(api_replay_rows_by_date[date])
        api_replay_count += len(api_replay_rows_by_date[date])
        print(f"✓ API/Streaming Replay {date_str}: {len(api_replay_rows_by_date[date])} rows → {daily_file}")
    
    # Report other rows (outside date ranges)
    if other_rows:
        other_file = os.path.join(output_dir, "other_dates.csv")
        with open(other_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(other_rows)
        print(f"⚠ Other dates: {len(other_rows)} rows → {other_file}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Batch Backfill: {len(batch_backfill_rows)} rows (1 file)")
    print(f"  Incremental Batch: {incremental_count} rows ({len(incremental_rows_by_date)} files)")
    print(f"  API/Streaming Replay: {api_replay_count} rows ({len(api_replay_rows_by_date)} files)")
    print(f"  Other dates: {len(other_rows)} rows")
    print(f"  Total: {len(batch_backfill_rows) + incremental_count + api_replay_count + len(other_rows)} rows")
    print("=" * 60)
    
    # Print directory structure
    print("\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── batch_backfill/")
    print(f"    │   └── batch_backfill_2023-11-26_to_2023-12-19.csv")
    print(f"    ├── incremental_batch/")
    print(f"    │   ├── incremental_2023-12-20.csv")
    print(f"    │   ├── incremental_2023-12-21.csv")
    print(f"    │   └── ... (daily files)")
    print(f"    └── api_streaming_replay/")
    print(f"        ├── api_replay_2023-12-24.csv")
    print(f"        ├── api_replay_2023-12-25.csv")
    print(f"        └── api_replay_2023-12-26.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Split CSV data into different ingestion patterns by date ranges'
    )
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for split files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    split_data_by_date_ranges(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
