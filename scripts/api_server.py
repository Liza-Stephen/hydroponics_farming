#!/usr/bin/env python3
"""
Simple API server to receive sensor data for ingestion simulation.

This server accepts POST requests with sensor data and writes them to
a staging area for batch ingestion.

Usage:
    python scripts/api_server.py --port 8000 --output-dir api_data/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Global variables
output_dir = None
records_buffer = []
buffer_size = 100  # Flush buffer after N records


def write_record_to_file(data: dict, output_dir: str):
    """Write a single record to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sensor_data_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


def flush_buffer(output_dir: str):
    """Flush buffered records to a single file."""
    if not records_buffer:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"sensor_batch_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump({"records": records_buffer.copy()}, f, indent=2)
    
    records_buffer.clear()
    return filepath


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
                filepath = flush_buffer(output_dir)
                print(f"Flushed {buffer_size} records to {filepath}")
            
            return jsonify({
                "status": "received",
                "records_count": len(records),
                "buffer_size": len(records_buffer)
            }), 200
        else:
            # Single record
            filepath = write_record_to_file(data, output_dir)
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
        filepath = flush_buffer(output_dir)
        return jsonify({"status": "flushed", "file": filepath}), 200
    return jsonify({"status": "buffer_empty"}), 200


def main():
    global output_dir
    
    parser = argparse.ArgumentParser(description='API server for sensor data ingestion')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Server port')
    parser.add_argument('--output-dir', '-o', default='api_data', help='Output directory for received data')
    parser.add_argument('--buffer-size', '-b', type=int, default=100, help='Buffer size before flushing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    global buffer_size
    buffer_size = args.buffer_size
    
    print(f"Starting API server on port {args.port}")
    print(f"Output directory: {output_dir}")
    print(f"Buffer size: {buffer_size}")
    print(f"API endpoint: http://localhost:{args.port}/api/sensor-data")
    print("-" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=True)


if __name__ == "__main__":
    main()
