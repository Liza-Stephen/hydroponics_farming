#!/usr/bin/env python3
"""
Simple API server to receive sensor data for ingestion simulation.

This server accepts POST requests with sensor data and writes them to
local files or directly to S3 for batch ingestion.

Usage:
    # Local file storage:
    python scripts/api_server.py --port 8000 --output-dir api_data/
    
    # S3 storage:
    python scripts/api_server.py --port 8000 --s3-bucket your-bucket --s3-prefix bronze/api_data/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Global variables
output_dir = None
s3_bucket = None
s3_prefix = None
s3_client = None
records_buffer = []
buffer_size = 100  # Flush buffer after N records


def write_record_to_file(data: dict, output_dir: str):
    """Write a single record to a local JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sensor_data_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


def write_record_to_s3(data: dict, s3_bucket: str, s3_prefix: str, s3_client):
    """Write a single record to S3 as JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sensor_data_{timestamp}.json"
    s3_key = f"{s3_prefix.rstrip('/')}/{filename}" if s3_prefix else filename
    
    try:
        json_data = json.dumps(data, indent=2)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        return f"s3://{s3_bucket}/{s3_key}"
    except (ClientError, NoCredentialsError) as e:
        print(f"Error writing to S3: {e}")
        raise


def write_record(data: dict):
    """Write a single record to local file or S3."""
    if s3_bucket and s3_client:
        return write_record_to_s3(data, s3_bucket, s3_prefix, s3_client)
    elif output_dir:
        return write_record_to_file(data, output_dir)
    else:
        raise ValueError("No output destination configured (neither output_dir nor S3)")


def flush_buffer_to_file(output_dir: str):
    """Flush buffered records to a local file."""
    if not records_buffer:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sensor_batch_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump({"records": records_buffer.copy()}, f, indent=2)
    
    records_buffer.clear()
    return filepath


def flush_buffer_to_s3(s3_bucket: str, s3_prefix: str, s3_client):
    """Flush buffered records to S3 as a single JSON file."""
    if not records_buffer:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sensor_batch_{timestamp}.json"
    s3_key = f"{s3_prefix.rstrip('/')}/{filename}" if s3_prefix else filename
    
    try:
        json_data = json.dumps({"records": records_buffer.copy()}, indent=2)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        records_buffer.clear()
        return f"s3://{s3_bucket}/{s3_key}"
    except (ClientError, NoCredentialsError) as e:
        print(f"Error writing to S3: {e}")
        raise


def flush_buffer():
    """Flush buffered records to local file or S3."""
    if s3_bucket and s3_client:
        return flush_buffer_to_s3(s3_bucket, s3_prefix, s3_client)
    elif output_dir:
        return flush_buffer_to_file(output_dir)
    else:
        raise ValueError("No output destination configured (neither output_dir nor S3)")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/api/sensor-data', methods=['POST'])
def receive_sensor_data():
    """Receive sensor data from API."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Handle batch or single record
        if "records" in data:
            # Batch of records
            records = data["records"]
            records_buffer.extend(records)
            
            if len(records_buffer) >= buffer_size:
                filepath = flush_buffer()
                print(f"Flushed {buffer_size} records to {filepath}")
            
            return jsonify({
                "status": "received",
                "records_count": len(records),
                "buffer_size": len(records_buffer)
            }), 200
        else:
            # Single record
            filepath = write_record(data)
            print(f"Received record: id={data.get('id', 'N/A')} -> {filepath}")
            
            return jsonify({
                "status": "received",
                "file": filepath
            }), 200
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/flush', methods=['POST'])
def flush():
    """Manually flush the buffer."""
    if records_buffer:
        filepath = flush_buffer()
        return jsonify({"status": "flushed", "file": filepath}), 200
    return jsonify({"status": "buffer_empty"}), 200


def main():
    global output_dir, s3_bucket, s3_prefix, s3_client
    
    parser = argparse.ArgumentParser(description='API server for sensor data ingestion')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Server port')
    parser.add_argument('--output-dir', '-o', help='Output directory for received data (local storage)')
    parser.add_argument('--s3-bucket', help='S3 bucket name for storing data')
    parser.add_argument('--s3-prefix', default='bronze/api_data/', help='S3 prefix/path for storing data')
    parser.add_argument('--buffer-size', '-b', type=int, default=100, help='Buffer size before flushing')
    
    args = parser.parse_args()
    
    # Validate configuration
    if not args.output_dir and not args.s3_bucket:
        print("Error: Either --output-dir or --s3-bucket must be specified")
        return 1
    
    if args.output_dir and args.s3_bucket:
        print("Warning: Both --output-dir and --s3-bucket specified. Using S3 (S3 takes precedence)")
        args.output_dir = None
    
    # Setup local file storage
    if args.output_dir:
        output_dir = args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Setup S3 storage
    if args.s3_bucket:
        if not BOTO3_AVAILABLE:
            print("Error: boto3 is required for S3 storage. Install with: pip install boto3")
            return 1
        
        s3_bucket = args.s3_bucket
        s3_prefix = args.s3_prefix
        
        try:
            s3_client = boto3.client('s3')
            # Test S3 connection
            s3_client.head_bucket(Bucket=s3_bucket)
            print(f"S3 bucket: s3://{s3_bucket}/{s3_prefix}")
            print("✓ S3 connection successful")
        except NoCredentialsError:
            print("Error: AWS credentials not found. Configure with 'aws configure' or set environment variables")
            return 1
        except ClientError as e:
            print(f"Error accessing S3 bucket: {e}")
            return 1
    
    global buffer_size
    buffer_size = args.buffer_size
    
    print(f"Starting API server on port {args.port}")
    print(f"Buffer size: {buffer_size}")
    print(f"API endpoint: http://localhost:{args.port}/api/sensor-data")
    print("-" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
