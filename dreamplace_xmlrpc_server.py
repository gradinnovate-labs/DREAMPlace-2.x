#!/usr/bin/env python3
"""
DREAMPlace XML-RPC Evaluation Server

Receives evaluation requests, creates task files, calls evaluator.py,
and returns results to clients.
"""

import json
import os
import sys
import subprocess
import tempfile
import traceback
from xmlrpc.server import SimpleXMLRPCServer
from typing import Dict, Any, Optional

# Use current directory for imports (evaluator.py in same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluator


class DREAMPlaceEvalService:
    """XML-RPC service for DREAMPlace evaluation"""

    def __init__(self, tmp_dir: Optional[str] = None):
        """
        Initialize the evaluation service

        Args:
            tmp_dir: Directory for task files (default: system temp)
        """
        self.tmp_dir = tmp_dir or os.path.join(os.path.dirname(__file__), 'tmp')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.evaluator = evaluator.DREAMPlaceEvaluator()

    def evaluate(self, task_id: str, config_file: str, params: Dict[str, Any],
                 detailed: bool = True, timeout: int = 3600,
                 sample_rate: int = 200) -> Dict[str, Any]:
        """
        Evaluate DREAMPlace with given parameters

        Args:
            task_id: Unique task identifier
            config_file: Path to base configuration file
            params: Dictionary of parameters to override (flattened format)
            detailed: Return detailed analysis (default: True)
            timeout: Timeout in seconds (default: 3600)
            sample_rate: Sampling rate for optimization trace (default: 200)

        Returns:
            Dictionary with evaluation results
        """
        print(f"[Task {task_id}] Received evaluation request")
        print(f"[Task {task_id}] Params: {params}")

        # Create task file
        task_file = self._create_task_file(task_id, config_file, params)

        try:
            # Run evaluation (evaluator will handle hierarchical mapping)
            print(f"[Task {task_id}] Running evaluator...")
            result = self.evaluator.evaluate(
                params=params,
                config_file=config_file,
                return_detailed_log=detailed,
                timeout=timeout,
                sample_rate=sample_rate
            )

            print(f"[Task {task_id}] Evaluation completed: HPWL={result.get('final_hpwl')}")

            # Clean up task file
            if os.path.exists(task_file):
                os.remove(task_file)

            return result

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            print(f"[Task {task_id}] {error_msg}")
            traceback.print_exc()

            # Clean up task file on error
            if os.path.exists(task_file):
                os.remove(task_file)

            return {
                "success": False,
                "error_message": error_msg,
                "task_id": task_id
            }

    def _create_task_file(self, task_id: str, config_file: str,
                         params: Dict[str, Any]) -> str:
        """
        Create task JSON file for logging/debugging

        Args:
            task_id: Task identifier
            config_file: Base configuration file
            params: Parameters

        Returns:
            Path to created task file
        """
        task_data = {
            "task_id": task_id,
            "config_file": config_file,
            "params": params,
            "timestamp": __import__('time').time()
        }

        task_file = os.path.join(self.tmp_dir, f"task_{task_id}.json")

        with open(task_file, 'w') as f:
            json.dump(task_data, f, indent=2)

        print(f"[Task {task_id}] Task file created: {task_file}")
        return task_file

    def ping(self) -> str:
        """Health check endpoint"""
        return "DREAMPlace XML-RPC Server is running"

    def get_supported_params(self) -> Dict[str, Any]:
        """Get information about supported parameters"""
        return {
            "num_bins_x": {"type": "choice", "values": [256, 512, 1024, 2048]},
            "num_bins_y": {"type": "choice", "values": [256, 512, 1024, 2048]},
            "optimizer": {"type": "choice", "values": ["adam", "nesterov"]},
            "wirelength": {"type": "choice", "values": ["weighted_average", "logsumexp"]},
            "learning_rate": {"type": "uniform", "min": 0.0001, "max": 0.01},
            "learning_rate_decay": {"type": "uniform", "min": 0.99, "max": 1.0},
            "density_weight": {"type": "uniform", "min": 0.000001, "max": 1.0},
            "target_density": {"type": "uniform", "min": 0.8, "max": 1.0},
            "gamma": {"type": "uniform", "min": 1.5, "max": 6.5},
            "gp_noise_ratio": {"type": "uniform", "min": 0.005, "max": 0.1}
        }


def main():
    """Start the XML-RPC server"""
    import argparse

    parser = argparse.ArgumentParser(description="DREAMPlace XML-RPC Evaluation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tmp-dir", help="Directory for task files")
    args = parser.parse_args()

    # Create service
    service = DREAMPlaceEvalService(tmp_dir=args.tmp_dir)

    # Create server
    server = SimpleXMLRPCServer((args.host, args.port), allow_none=True)
    server.register_introspection_functions()

    # Register service
    server.register_instance(service)

    print(f"Starting DREAMPlace XML-RPC Server on {args.host}:{args.port}")
    print(f"Task files will be stored in: {service.tmp_dir}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")


if __name__ == "__main__":
    main()
