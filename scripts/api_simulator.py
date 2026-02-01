#!/usr/bin/env python3
"""
Simulate API-based data ingestion by sending sensor data via HTTP POST requests.

This script reads from CSV and sends data to an API endpoint, simulating
real-time IoT sensor data ingestion.

Usage:
    # Start API server first (in another terminal):
    python scripts/api_server.py
    
    # Then run simulator:
    python scripts/api_simulator.py --input raw_data/iot_data_raw.csv --api-url http://localhost:8000/api/sensor-data
    python scripts/api_simulator.py --input raw_data/iot_data_raw.csv --api-url http://localhost:8000/api/sensor-data --delay 1 --batch-size 10
"""

import argparse
import csv
import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, List


def send_sensor_data(api_url: str, data: Dict) -> bool:
    """
    Send sensor data to API endpoint.
    
    Args:
        api_url: API endpoint URL
        data: Sensor data dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            api_url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending data: {e}")
        return False


def csv_row_to_json(row: List[str], headers: List[str]) -> Dict:
    """Convert CSV row to JSON format."""
    return dict(zip(headers, row))


def simulate_api_ingestion(
    input_file: str,
    api_url: str,
    delay: float = 0.1,
    batch_size: int = 1,
    start_from: int = 0,
    max_records: int = None
):
    """
    Simulate API ingestion by reading CSV and sending data to API.
    
    Args:
        input_file: Path to input CSV file
        api_url: API endpoint URL
        delay: Delay between requests (seconds)
        batch_size: Number of records to send per request
        start_from: Row number to start from (0-indexed, excluding header)
        max_records: Maximum number of records to send (None for all)
    """
    print(f"Starting API ingestion simulation...")
    print(f"API URL: {api_url}")
    print(f"Delay: {delay}s between requests")
    print(f"Batch size: {batch_size} records per request")
    print(f"Starting from row: {start_from}")
    if max_records:
        print(f"Max records: {max_records}")
    print("-" * 60)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read header
        
        # Skip to start position
        for _ in range(start_from):
            next(reader, None)
        
        batch = []
        sent_count = 0
        success_count = 0
        error_count = 0
        
        try:
            for row in reader:
                if max_records and sent_count >= max_records:
                    break
                
                # Convert row to JSON
                data = csv_row_to_json(row, headers)
                batch.append(data)
                
                # Send batch when full
                if len(batch) >= batch_size:
                    if batch_size == 1:
                        # Send single record
                        success = send_sensor_data(api_url, batch[0])
                        if success:
                            success_count += 1
                            print(f"✓ Sent record {sent_count + 1}: id={batch[0].get('id', 'N/A')}")
                        else:
                            error_count += 1
                    else:
                        # Send batch
                        success = send_sensor_data(api_url, {"records": batch})
                        if success:
                            success_count += len(batch)
                            print(f"✓ Sent batch: {len(batch)} records (total: {sent_count + len(batch)})")
                        else:
                            error_count += len(batch)
                    
                    sent_count += len(batch)
                    batch = []
                    
                    # Delay between requests
                    if delay > 0:
                        time.sleep(delay)
            
            # Send remaining batch
            if batch:
                if batch_size == 1:
                    success = send_sensor_data(api_url, batch[0])
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                else:
                    success = send_sensor_data(api_url, {"records": batch})
                    if success:
                        success_count += len(batch)
                    else:
                        error_count += len(batch)
                sent_count += len(batch)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        print("\n" + "=" * 60)
        print(f"Simulation complete!")
        print(f"Total sent: {sent_count}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Simulate API-based data ingestion')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--api-url', '-u', required=True, help='API endpoint URL')
    parser.add_argument('--delay', '-d', type=float, default=0.1, help='Delay between requests (seconds)')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Number of records per request')
    parser.add_argument('--start-from', type=int, default=0, help='Row number to start from (0-indexed)')
    parser.add_argument('--max-records', '-m', type=int, help='Maximum number of records to send')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    simulate_api_ingestion(
        args.input,
        args.api_url,
        args.delay,
        args.batch_size,
        args.start_from,
        args.max_records
    )
    return 0


if __name__ == "__main__":
    import os
    import sys
    exit(main())
