"""
Test script to generate basic visualizations for a single subject with a few gamma values
(without lambda history tracking)
"""

from work_please import FairnessElicitationAlgorithm
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_trajectory(results, gamma_values=None, title="Algorithm Trajectory", 
                   save_path=None):
    """Plot trajectory for a single subject with multiple gamma values"""
    if gamma_values is None:
        gamma_values = sorted(list(results.keys()))
    
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.xlabel("error(t)")
    plt.ylabel("max(violation(t))")
    
    # Add horizontal lines at 0.1 intervals
    for y in np.arange(0, 1.1, 0.1):
        plt.axhline(y=y, color='r', linestyle='-', alpha=0.3)
    
    # Plot trajectory for each gamma
    for gamma in gamma_values:
        if gamma in results:
            errors = results[gamma]['errors']
            violations = results[gamma]['max_violations']
            plt.plot(errors, violations, label=f"γ = {gamma}")
            
            # Mark start and end points
            plt.scatter(errors[0], violations[0], color='green', s=50, 
                        marker='o', label=f"Start γ={gamma}" if gamma == gamma_values[0] else "")
            plt.scatter(errors[-1], violations[-1], color='red', s=50, 
                        marker='x', label=f"End γ={gamma}" if gamma == gamma_values[0] else "")
    
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

def plot_pareto_curves(results, title="Pareto Curve: Error vs. Fairness Violation", 
                      save_path=None):
    """Plot the Pareto curve for a subject (error vs fairness violation)"""
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Maximum Fairness Violation")
    
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
    
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Pareto curve to {save_path}")
    else:
        plt.show()

def test_visualizations():
    # Configuration
    data_path = "data/processed/compas_train.parquet"
    constraint_path = "constraint_sets/lenient/binary_personas/constraint_sets.json"
    output_dir = "results/test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize algorithm with test parameters
    algorithm = FairnessElicitationAlgorithm(
        data_path=data_path,
        constraint_sets_path=constraint_path,
        time_horizon=100,  # Reduced for faster testing
        C_lambda=1.0,
        C_tau=1.0
    )
    
    # Run for just a few gamma values
    gamma_values = [0.0, 0.3, 0.5, 0.7]
    results = algorithm.run_individual_subject_analysis(subject_id=437, gamma_values=gamma_values)
    
    # Generate basic visualizations
    plot_trajectory(
        results, 
        title="Test: Single-Subject Trajectory",
        save_path=os.path.join(output_dir, "test_trajectory.png")
    )
    
    plot_pareto_curves(
        results,
        save_path=os.path.join(output_dir, "test_pareto.png")
    )
    
    print(f"Test visualizations saved to {output_dir}")

if __name__ == "__main__":
    test_visualizations()