#!/usr/bin/env python3
"""
Batch Fairness Algorithm Runner and Visualizer
----------------------------------------------
This script:
1. Runs the Fairness Elicitation Algorithm for multiple judge IDs
2. Visualizes results for each judge and gamma value
3. Organizes outputs by judge ID

Usage: python batch_fairness.py --constraint_path /path/to/constraint_sets.json --data_path /path/to/data.parquet --out_dir /path/to/output --gamma_values 0.0,0.1,0.2,0.3,0.4,0.5

Can be run in parallel using SLURM with array jobs.
"""

import argparse
import glob
import os
import pickle
import sys
import traceback
import json
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import MaxNLocator
import time

# Import the FairnessElicitationAlgorithm from river_fairness
try:
    from river_fairness import FairnessElicitationAlgorithm
except ImportError:
    print("Error: river_fairness module not found. Make sure it's in your PYTHONPATH.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch Fairness Algorithm Runner and Visualizer")
    parser.add_argument("--constraint_path", required=True, help="Path to the constraint sets JSON file")
    parser.add_argument("--data_path", required=True, help="Path to the COMPAS data parquet file")
    parser.add_argument("--out_dir", required=True, help="Output directory for results and visualizations")
    parser.add_argument("--gamma_values", default="0.0,0.1,0.2,0.3,0.4,0.5", 
                        help="Comma-separated list of gamma values to use")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to run algorithm")
    parser.add_argument("--c_lambda", type=float, default=1.0, help="C_lambda parameter")
    parser.add_argument("--c_tau", type=float, default=1.0, help="C_tau parameter")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="Number of judges to process in one batch (for SLURM array)")
    parser.add_argument("--batch_id", type=int, default=None, 
                        help="Batch ID for SLURM array jobs (if not set, process all judges)")
    parser.add_argument("--visualization_only", action="store_true", 
                        help="Only generate visualizations, don't run the algorithm")
    parser.add_argument("--judge_start", type=int, default=1, 
                        help="First judge ID to process")
    parser.add_argument("--judge_end", type=int, default=999, 
                        help="Last judge ID to process")
    
    return parser.parse_args()

def run_fairness_algorithm(judge_id, constraint_path, data_path, output_dir, gamma_values, iterations, c_lambda, c_tau):
    """Run fairness algorithm for a single judge/persona with multiple gamma values"""
    try:
        # Create output directories
        persona_name = f"judge_{judge_id}"
        results_dir = os.path.join(output_dir, "results", persona_name)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Starting algorithm for {persona_name}")
        start_time = time.time()
        
        # Initialize the algorithm
        algorithm = FairnessElicitationAlgorithm(
            data_path=data_path,
            constraint_sets_path=constraint_path,
            time_horizon=iterations,
            C_lambda=c_lambda,
            C_tau=c_tau
        )
        
        # Load constraints specifically for this judge
        algorithm.load_constraint_sets(judge_id=str(judge_id))
        
        # Parse gamma values
        gamma_list = [float(g) for g in gamma_values.split(',')]
        
        # Run the algorithm with specified gamma values
        results = algorithm.run(gamma_values=gamma_list)
        
        # Save results to file
        results_path = os.path.join(results_dir, f"results_all_gammas.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        end_time = time.time()
        print(f"Completed algorithm for persona: {persona_name} in {end_time - start_time:.2f} seconds")
        
        # Create summary statistics for quick reference
        summary = {
            "persona": persona_name,
            "iterations": iterations,
            "gamma_values": gamma_list,
            "results": {}
        }
        
        for gamma in gamma_list:
            summary["results"][gamma] = {
                "final_error": results[gamma]["final_error"],
                "final_fairness_violation": results[gamma]["final_fairness_violation"],
                "final_max_violation": results[gamma]["final_max_violation"]
            }
        
        # Save summary
        with open(os.path.join(results_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return results_path
        
    except Exception as e:
        print(f"Error processing persona {persona_path}: {e}")
        traceback.print_exc()
        return None

def visualize_results(results_path, output_dir):
    """Visualize the results from a pickle file"""
    try:
        # Load results
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract persona name from path
        parts = os.path.basename(os.path.dirname(results_path)).split('_')
        persona_name = parts[-1] if len(parts) > 1 else parts[0]
        
        # Create visualization directory
        vis_dir = os.path.join(output_dir, "visualizations", persona_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create a summary plot for all gamma values
        create_summary_plots(results, persona_name, vis_dir)
        
        # Create individual plots for each gamma value
        for gamma, result in results.items():
            create_individual_plots(gamma, result, persona_name, vis_dir)
        
        print(f"Generated visualizations for persona: {persona_name}")
        return True
    
    except Exception as e:
        print(f"Error generating visualizations for {results_path}: {e}")
        traceback.print_exc()
        return False

def create_summary_plots(results, persona_name, output_dir):
    """Create summary plots for all gamma values"""
    
    # 1. Pareto curve (Error vs. Max Violation)
    plt.figure(figsize=(12, 8))
    
    # Extract final errors and violations for each gamma
    gammas = sorted(list(results.keys()))
    errors = [results[gamma]['final_error'] for gamma in gammas]
    violations = [results[gamma]['final_max_violation'] for gamma in gammas]
    
    # Plot the Pareto curve with points
    plt.plot(errors, violations, 'o-', markersize=8)
    
    # Add gamma labels to points
    for i, gamma in enumerate(gammas):
        plt.annotate(f"γ={gamma}", 
                    (errors[i], violations[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.xlabel('Error')
    plt.ylabel('Maximum Fairness Violation')
    plt.title(f'Pareto Curve for Persona: {persona_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_curve.png"))
    plt.close()
    
    # 2. Convergence comparison for all gamma values
    plt.figure(figsize=(14, 10))
    
    # Add subplots for error and violation
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    # Plot error convergence for each gamma
    for gamma in gammas:
        iterations = list(range(1, len(results[gamma]['errors']) + 1))
        ax1.plot(iterations, results[gamma]['errors'], label=f"γ={gamma}")
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error')
    ax1.set_title(f'Error Convergence for Different γ Values (Persona: {persona_name})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot max violation convergence for each gamma
    for gamma in gammas:
        iterations = list(range(1, len(results[gamma]['max_violations']) + 1))
        ax2.plot(iterations, results[gamma]['max_violations'], label=f"γ={gamma}")
        # Add horizontal line at gamma
        ax2.axhline(y=gamma, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Fairness Violation')
    ax2.set_title(f'Fairness Violation Convergence for Different γ Values')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"))
    plt.close()
    
    # 3. Trajectory comparison (multiple trajectories in one plot)
    plt.figure(figsize=(12, 8))
    
    for gamma in gammas:
        # Plot trajectory
        plt.plot(results[gamma]['errors'], results[gamma]['max_violations'], 
                label=f"γ={gamma}", alpha=0.7)
        
        # Mark start and end points
        plt.scatter(results[gamma]['errors'][0], results[gamma]['max_violations'][0], 
                   marker='o', s=50)
        plt.scatter(results[gamma]['errors'][-1], results[gamma]['max_violations'][-1], 
                   marker='x', s=50)
    
    plt.xlabel('Error')
    plt.ylabel('Max Fairness Violation')
    plt.title(f'Algorithm Trajectories for Different γ Values (Persona: {persona_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"))
    plt.close()

def create_individual_plots(gamma, result, persona_name, output_dir):
    """Create detailed plots for a single gamma value"""
    
    # 1. Convergence plot - Error and Max Violation vs Iteration
    iterations = list(range(1, len(result['errors']) + 1))
    
    plt.figure(figsize=(12, 6))
    
    # Plot error progression
    ax1 = plt.subplot(111)
    ax1.plot(iterations, result['errors'], 'b-', label='Error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlim(0, len(iterations) + 1)
    
    # Create second y-axis for fairness violations
    ax2 = ax1.twinx()
    ax2.plot(iterations, result['max_violations'], 'r-', label='Max Fairness Violation')
    ax2.set_ylabel('Max Fairness Violation', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add a horizontal line at gamma
    ax2.axhline(y=gamma, color='r', linestyle='--', alpha=0.7, label=f'γ={gamma}')
    
    # Add title and grid
    plt.title(f'Convergence Plot for γ={gamma} (Persona: {persona_name})')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"convergence_gamma_{gamma}.png"))
    plt.close()
    
    # 2. Lambda values distribution
    lambda_final = result.get('lambda_final', {})
    if lambda_final:
        # Only include non-zero lambdas
        non_zero_lambdas = {k: v for k, v in lambda_final.items() if v > 0.001}
        
        if non_zero_lambdas:
            plt.figure(figsize=(12, 6))
            
            # Sort lambda values
            sorted_lambdas = sorted(non_zero_lambdas.items(), key=lambda x: x[1], reverse=True)
            pairs = [str(pair) for pair, _ in sorted_lambdas[:20]]  # Take top 20 for readability
            values = [value for _, value in sorted_lambdas[:20]]
            
            # Create bar chart
            plt.bar(range(len(pairs)), values, color='skyblue')
            plt.xticks(range(len(pairs)), pairs, rotation=90)
            plt.xlabel('Constraint Pairs')
            plt.ylabel('Lambda Value')
            plt.title(f'Top Lambda Values for γ={gamma} (Persona: {persona_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"lambdas_gamma_{gamma}.png"))
            plt.close()
    
    # 3. Trajectory plot (Error vs Max Violation)
    plt.figure(figsize=(10, 8))
    plt.scatter(result['errors'], result['max_violations'], c=iterations, cmap='viridis', 
               alpha=0.7, s=30)
    
    # Mark start and end points
    plt.scatter(result['errors'][0], result['max_violations'][0], color='green', s=100, 
               marker='o', label='Start')
    plt.scatter(result['errors'][-1], result['max_violations'][-1], color='red', s=100, 
               marker='x', label='End')
    
    # Add arrow to show direction if there are enough iterations
    if len(result['errors']) > 10:
        mid_idx = len(result['errors']) // 2
        plt.annotate('', 
                    xy=(result['errors'][mid_idx+1], result['max_violations'][mid_idx+1]),
                    xytext=(result['errors'][mid_idx], result['max_violations'][mid_idx]),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    )
    
    plt.colorbar(label='Iteration')
    plt.xlabel('Error')
    plt.ylabel('Max Fairness Violation')
    plt.title(f'Algorithm Trajectory for γ={gamma} (Persona: {persona_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trajectory_gamma_{gamma}.png"))
    plt.close()

def process_judge(judge_id, constraint_path, args):
    """Process a single judge/persona: run algorithm and visualize results"""
    try:
        persona_name = f"judge_{judge_id}"
        print(f"Processing {persona_name}")
        
        results_dir = os.path.join(args.out_dir, "results", persona_name)
        results_path = os.path.join(results_dir, "results_all_gammas.pkl")
        
        # Run algorithm if needed
        if not args.visualization_only and (not os.path.exists(results_path) or 
                                          os.path.getsize(results_path) == 0):
            results_path = run_fairness_algorithm(
                judge_id=judge_id,
                constraint_path=constraint_path,
                data_path=args.data_path,
                output_dir=args.out_dir, 
                gamma_values=args.gamma_values,
                iterations=args.iterations,
                c_lambda=args.c_lambda,
                c_tau=args.c_tau
            )
        
        # Visualize results if available
        if results_path and os.path.exists(results_path):
            visualize_results(results_path, args.out_dir)
        elif args.visualization_only:
            print(f"Warning: Results file not found for persona: {persona_name}")
        
        return persona_name
    
    except Exception as e:
        print(f"Error processing persona {persona_path}: {e}")
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "visualizations"), exist_ok=True)
    
    # Check if constraint file exists
    if not os.path.exists(args.constraint_path):
        print(f"Error: Constraint file not found at {args.constraint_path}")
        sys.exit(1)
    
    # Generate list of all judge IDs to process
    all_judges = list(range(args.judge_start, args.judge_end + 1))
    print(f"Will process judges {args.judge_start} to {args.judge_end}")
    
    # If batch_id is provided, only process that batch
    if args.batch_id is not None:
        start_idx = args.batch_id * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(all_judges))
        
        if start_idx >= len(all_judges):
            print(f"Error: Batch {args.batch_id} is out of range. There are only {len(all_judges)} judges.")
            sys.exit(1)
        
        judges_to_process = all_judges[start_idx:end_idx]
        print(f"Processing batch {args.batch_id} ({len(judges_to_process)} judges)")
    else:
        judges_to_process = all_judges
    
    # Process judges sequentially (parallel processing can be done via SLURM array jobs)
    for judge_id in judges_to_process:
        process_judge(judge_id, args.constraint_path, args)

    print(f"Completed processing {len(judges_to_process)} judges")
    
    # Create an index.html file with links to all visualizations
    create_index_html(args.out_dir)

def create_index_html(output_dir):
    """Create an index.html file with links to all persona visualizations"""
    vis_dir = os.path.join(output_dir, "visualizations")
    personas = sorted([d for d in os.listdir(vis_dir) if os.path.isdir(os.path.join(vis_dir, d))])
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fairness Algorithm Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .persona-list { columns: 3; -webkit-columns: 3; -moz-columns: 3; }
            .persona-item { margin-bottom: 10px; break-inside: avoid; }
        </style>
    </head>
    <body>
        <h1>Fairness Algorithm Results</h1>
        <p>Click on a persona to view its visualizations:</p>
        <div class="persona-list">
    """
    
    for persona in personas:
        html_content += f'<div class="persona-item"><a href="visualizations/{persona}/index.html">{persona}</a></div>\n'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write main index.html
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(html_content)
    
    # Create individual index.html files for each persona
    for persona in personas:
        persona_dir = os.path.join(vis_dir, persona)
        image_files = [f for f in os.listdir(persona_dir) if f.endswith('.png')]
        
        # Group images by type
        summary_images = [img for img in image_files if 'comparison' in img or 'pareto' in img]
        gamma_images = sorted([img for img in image_files if img not in summary_images], 
                             key=lambda x: float(x.split('gamma_')[1].split('.png')[0]))
        
        persona_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Results for Persona: {persona}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .image-container {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
                .back-link {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="back-link"><a href="../../index.html">← Back to All Personas</a></div>
            <h1>Results for Persona: {persona}</h1>
            
            <h2>Summary Visualizations</h2>
        """
        
        for img in summary_images:
            persona_html += f"""
            <div class="image-container">
                <h3>{img.split('.')[0].replace('_', ' ').title()}</h3>
                <img src="{img}" alt="{img}">
            </div>
            """
        
        persona_html += "<h2>Individual Gamma Results</h2>"
        
        # Group images by gamma
        gamma_values = sorted(list(set([float(img.split('gamma_')[1].split('.png')[0]) 
                                       for img in gamma_images if 'gamma_' in img])))
        
        for gamma in gamma_values:
            persona_html += f"<h3>Gamma = {gamma}</h3>"
            gamma_imgs = [img for img in gamma_images if f'gamma_{gamma}' in img]
            
            for img in gamma_imgs:
                persona_html += f"""
                <div class="image-container">
                    <h4>{img.split('gamma_')[0].replace('_', ' ').title()}</h4>
                    <img src="{img}" alt="{img}">
                </div>
                """
        
        persona_html += """
        </body>
        </html>
        """
        
        # Write persona index.html
        with open(os.path.join(persona_dir, "index.html"), 'w') as f:
            f.write(persona_html)
    
    print(f"Created index.html files")

if __name__ == "__main__":
    main()