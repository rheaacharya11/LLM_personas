
#!/usr/bin/env python3
"""
Pickle Visualizer for Fairness Algorithm Results
------------------------------------------------
This script loads and visualizes the results saved by the fairness algorithm.

Usage: python visualize_results.py results_gamma_0.3.pkl
"""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

def load_pickle(file_path):
    """Load pickle file and return its contents"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def visualize_results(results, output_dir='visualizations'):
    """Visualize the results from the pickle file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract gamma values
    gamma_values = list(results.keys())
    
    for gamma in gamma_values:
        result = results[gamma]
        
        # Plot 1: Convergence plot - Error and Max Violation vs Iteration
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
        plt.title(f'Convergence Plot for γ={gamma}')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_gamma_{gamma}.png')
        
        # Plot 2: Lambda values distribution
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
                plt.title(f'Top Lambda Values for γ={gamma}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/lambdas_gamma_{gamma}.png')
        
        # Plot 3: Trajectory plot (Error vs Max Violation)
        plt.figure(figsize=(10, 8))
        plt.scatter(result['errors'], result['max_violations'], c=iterations, cmap='viridis', 
                   alpha=0.7, s=30)
        
        # Mark start and end points
        plt.scatter(result['errors'][0], result['max_violations'][0], color='green', s=100, 
                   marker='o', label='Start')
        plt.scatter(result['errors'][-1], result['max_violations'][-1], color='red', s=100, 
                   marker='x', label='End')
        
        # Add arrow to show direction
        mid_idx = len(result['errors']) // 2
        plt.annotate('', 
                    xy=(result['errors'][mid_idx+1], result['max_violations'][mid_idx+1]),
                    xytext=(result['errors'][mid_idx], result['max_violations'][mid_idx]),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    )
        
        plt.colorbar(label='Iteration')
        plt.xlabel('Error')
        plt.ylabel('Max Fairness Violation')
        plt.title(f'Algorithm Trajectory for γ={gamma}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trajectory_gamma_{gamma}.png')
        
        print(f"Generated visualizations for γ={gamma} in '{output_dir}' directory")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <pickle_file_path>")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    if not os.path.exists(pickle_path):
        print(f"Error: File '{pickle_path}' not found")
        sys.exit(1)
    
    try:
        results = load_pickle(pickle_path)
        visualize_results(results)
        print(f"Successfully visualized results from {pickle_path}")
    except Exception as e:
        print(f"Error processing pickle file: {e}")
        import traceback
        traceback.print_exc()