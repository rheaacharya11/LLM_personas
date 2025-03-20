#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script allows you to easily run multiple experimental configurations by changing parameters.
"""
import os
import subprocess
import argparse
import time
import json
from datetime import datetime


def run_experiment(config):
    # Start time for this configuration
    start_time = time.time()
    
    # Build the command with all parameters
    cmd = ["python", "../../src/comparison_elicitation.py"]
    
    # Add all parameters
    for key, value in config.items():
        if key.startswith("_"):  # Skip metadata fields starting with _
            continue
            
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            else:
                cmd.append(f"--no_{key.replace('use_', '')}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Print command for debugging
    print(f"\nRunning experiment: {config.get('_name', 'unnamed')}")
    print(f"Command: {' '.join(cmd)}")
    
    # Create output directory if needed
    output_file = config.get('output')
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Experiment completed with exit code: {result.returncode}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        return False


def run_experimental_batch(configurations, sequential=True):
    batch_start_time = time.time()
    print(f"Starting batch of {len(configurations)} experiments at {datetime.now()}")
    
    results = []
    
    if sequential:
        # Run experiments one at a time
        for config in configurations:
            success = run_experiment(config)
            results.append((config.get('_name', 'unnamed'), success))
    else:
        # TODO: Implement parallel execution if needed
        print("Parallel execution not yet implemented, running sequentially")
        for config in configurations:
            success = run_experiment(config)
            results.append((config.get('_name', 'unnamed'), success))
    
    # Report overall results
    batch_elapsed_time = time.time() - batch_start_time
    print("\n=== Batch Execution Report ===")
    print(f"Total time: {batch_elapsed_time:.2f} seconds")
    print(f"Successful experiments: {sum(1 for _, success in results if success)}/{len(results)}")
    
    for name, success in results:
        status = "✓ Succeeded" if success else "✗ Failed"
        print(f"  - {name}: {status}")
    
    # Save execution report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": batch_elapsed_time,
        "configurations": [
            {
                "name": config.get('_name', 'unnamed'),
                "parameters": {k: v for k, v in config.items() if not k.startswith('_')},
                "success": success
            }
            for (name, success), config in zip(results, configurations)
        ]
    }
    
    with open("experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Execution report saved to experiment_report.json")


def main():
    parser = argparse.ArgumentParser(description="Run fairness elicitation experiments")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--config", type=str, help="Path to experiment configuration file")
    args = parser.parse_args()
    
    # Define experimental configurations
    if args.config:
        # Load configurations from file
        try:
            with open(args.config, 'r') as f:
                configurations = json.load(f)
            print(f"Loaded {len(configurations)} configurations from {args.config}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return
    else:
        # Use default configurations
        configurations = [
            # Binary judgment experiment (similar vs different)
            {
                "_name": "binary_judgment",
                "train_path": "../../data/processed/compas_train.parquet",
                "personas_path": "../../data/unique_personas.parquet",
                "output": "../results/binary_judgment.csv",
                "prompt_config": "../../prompts/binary_config.yaml",
                "pairs_per_persona": 10,  # Using fewer pairs for testing
                "use_personas": True,
                "prompt_type": "chain_of_thought",
                "judgment_type": "binary",
                "start_index": 0,
                "end_index": 5  # Use just a few personas for testing
            },
            
            # Three-option judgment experiment
            {
                "_name": "three_option_judgment",
                "train_path": "../../data/processed/compas_train.parquet",
                "personas_path": "../../data/unique_personas.parquet",
                "output": "../results/three_option_judgment.csv",
                "prompt_config": "../../prompts/three_option_config.yaml",
                "pairs_per_persona": 10,  # Using fewer pairs for testing
                "use_personas": True,
                "prompt_type": "chain_of_thought",
                "judgment_type": "three_option",
                "start_index": 0,
                "end_index": 5  # Use just a few personas for testing
            },
            
            # Without personas, just logical chain of thought
            {
                "_name": "no_personas",
                "train_path": "../../data/processed/compas_train.parquet",
                "personas_path": "../../data/unique_personas.parquet",
                "output": "../results/no_personas.csv",
                "prompt_config": "../../prompts/three_option_config.yaml",
                "pairs_per_persona": 10,
                "use_personas": False,
                "prompt_type": "chain_of_thought",
                "judgment_type": "three_option",
                "start_index": 0,
                "end_index": 5
            },
            
            # Default prompt (less structured reasoning)
            {
                "_name": "default_prompt",
                "train_path": "../../data/processed/compas_train.parquet",
                "personas_path": "../../data/unique_personas.parquet",
                "output": "../results/default_prompt.csv",
                "prompt_config": "../../prompts/three_option_config.yaml",
                "pairs_per_persona": 10,
                "use_personas": False,
                "prompt_type": "default",
                "judgment_type": "three_option",
                "start_index": 0,
                "end_index": 5
            }
        ]
    
    # Run the experiments
    run_experimental_batch(configurations, sequential=not args.parallel)


if __name__ == "__main__":
    main()