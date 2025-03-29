import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

class TrajectoryRecorder:
    """Helper class to save and visualize algorithm trajectory data"""
    
    def __init__(self, output_dir="results"):
        """Initialize with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_trajectory_data(self, results, prefix="subject"):
        """
        Save trajectory data for each gamma value to CSV files
        
        Args:
            results: Dictionary with gamma values as keys and results as values
            prefix: Prefix for filenames (e.g., "subject_1")
        """
        # Create trajectory directory
        trajectory_dir = os.path.join(self.output_dir, "trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # Save summary data
        summary_data = []
        for gamma, result in results.items():
            summary_data.append({
                'gamma': gamma,
                'final_error': result['final_error'],
                'final_fairness_violation': result['final_fairness_violation'],
                'final_max_violation': result['final_max_violation'],
                'iterations': len(result['errors'])
            })
            
            # Save detailed trajectory for this gamma
            trajectory_df = pd.DataFrame({
                'iteration': range(1, len(result['errors']) + 1),
                'error': result['errors'],
                'fairness_violation': result['fairness_violations'],
                'max_violation': result['max_violations']
            })
            
            # Save to CSV
            trajectory_file = os.path.join(trajectory_dir, f"{prefix}_gamma_{gamma}.csv")
            trajectory_df.to_csv(trajectory_file, index=False)
            print(f"Saved trajectory data to {trajectory_file}")
            
        # Save summary data
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, f"{prefix}_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary data to {summary_file}")
        
        # Save lambda values for analysis
        lambda_dir = os.path.join(self.output_dir, "lambda_values")
        os.makedirs(lambda_dir, exist_ok=True)
        
        for gamma, result in results.items():
            # Convert lambda dictionary to a serializable format
            lambda_dict = {str(pair): value for pair, value in result['lambda_final'].items()}
            lambda_file = os.path.join(lambda_dir, f"{prefix}_gamma_{gamma}_lambda.json")
            with open(lambda_file, 'w') as f:
                json.dump(lambda_dict, f, indent=2)
    
    def plot_single_trajectory(self, results, gamma_values=None, title="Algorithm Trajectory", 
                               save_path=None):
        """
        Plot trajectory for a single subject with multiple gamma values
        
        Args:
            results: Dictionary with gamma values as keys and results as values
            gamma_values: List of gamma values to plot (if None, plots all)
            title: Plot title
            save_path: Path to save the plot (if None, shows the plot)
        """
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
    
    def plot_pareto_curves(self, results, save_path=None):
        """
        Plot the Pareto curve for a subject (error vs fairness violation)
        
        Args:
            results: Results from running the algorithm
            save_path: Path to save the plot (if None, shows the plot)
        """
        plt.figure(figsize=(12, 8))
        plt.title("Pareto Curve: Error vs. Fairness Violation")
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
    
    def plot_lambda_evolution(self, results, gamma, top_n=5, save_path=None):
        """
        Plot the evolution of lambda values for the top constraint pairs
        
        Args:
            results: Results for a specific gamma value
            gamma: Gamma value to analyze
            top_n: Number of top constraint pairs to plot
            save_path: Path to save the plot (if None, shows the plot)
        """
        # Check if lambda history is available
        if 'lambda_history' not in results[gamma]:
            print("Lambda history not available. Update algorithm to track lambda values per iteration.")
            return
            
        lambda_history = results[gamma]['lambda_history']
        iterations = len(lambda_history)
        
        # Find the top_n pairs with highest final lambda values
        final_lambdas = lambda_history[-1]
        top_pairs = sorted(final_lambdas.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Evolution of λ Values for Top {top_n} Constraints (γ={gamma})")
        plt.xlabel("Iteration")
        plt.ylabel("λ Value")
        
        for pair, _ in top_pairs:
            values = [lambda_dict.get(pair, 0) for lambda_dict in lambda_history]
            plt.plot(range(1, iterations+1), values, label=f"Pair {pair}")
        
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved lambda evolution plot to {save_path}")
        else:
            plt.show()
    
    def plot_multiple_trajectories(self, subject_results, gamma=0.3, 
                                 title="Trajectory Comparison Across Subjects", 
                                 save_path=None):
        """
        Compare trajectories for multiple subjects at a specific gamma value
        
        Args:
            subject_results: Dictionary with subject IDs as keys and results as values
            gamma: Gamma value to compare (must be present in all subjects' results)
            title: Plot title
            save_path: Path to save the plot (if None, shows the plot)
        """
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.xlabel("error(t)")
        plt.ylabel("max(violation(t))")
        
        # Add horizontal lines at 0.1 intervals
        for y in np.arange(0, 1.1, 0.1):
            plt.axhline(y=y, color='r', linestyle='-', alpha=0.3)
        
        # Plot trajectory for each subject
        for subject_id, results in subject_results.items():
            if gamma in results:
                errors = results[gamma]['errors']
                violations = results[gamma]['max_violations']
                plt.plot(errors, violations, label=f"Subject {subject_id}")
        
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-subject trajectory plot to {save_path}")
        else:
            plt.show()
            
    def plot_multiple_pareto(self, subject_results, 
                            title="Variability of Subject Pareto Curves", 
                            save_path=None):
        """
        Plot Pareto curves for multiple subjects on the same graph
        
        Args:
            subject_results: Dictionary with subject IDs as keys and results as values
            title: Plot title
            save_path: Path to save the plot (if None, shows the plot)
        """
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.xlabel("error")
        plt.ylabel("max violation")
        
        # Plot Pareto curve for each subject
        for subject_id, results in subject_results.items():
            # Extract final errors and violations for each gamma
            gammas = sorted(list(results.keys()))
            errors = [results[gamma]['final_error'] for gamma in gammas]
            violations = [results[gamma]['final_max_violation'] for gamma in gammas]
            
            # Plot the Pareto curve
            plt.plot(errors, violations, label=f"Subject {subject_id}" if len(subject_results) <= 10 else None)
        
        if len(subject_results) <= 10:
            plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-subject Pareto curves to {save_path}")
        else:
            plt.show()