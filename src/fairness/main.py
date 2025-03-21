import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .utils import load_constraints, compute_constraint_weights, load_training_data
from .no_regret import NoRegretFairness

def plot_convergence(errors, fairness_violations, gamma, output_path=None):
    """
    Plot convergence of error and fairness violation over iterations.
    
    Args:
        errors: List of errors for each iteration
        fairness_violations: List of fairness violations for each iteration
        gamma: Gamma parameter used
        output_path: Optional path to save the plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    iters = np.arange(1, len(errors) + 1)
    
    # Plot error
    color = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Error', color=color)
    ax1.plot(iters, errors, color=color, label='Error')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for fairness violation
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Fairness Violation', color=color)
    ax2.plot(iters, fairness_violations, color=color, label='Fairness Violation')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=gamma, color='tab:orange', linestyle='--', label=f'Gamma = {gamma}')
    
    # Add title and legend
    plt.title(f'Convergence of No-Regret Algorithm (Gamma = {gamma})')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def plot_pareto_curve(results, output_path=None):
    """
    Plot Pareto curve of error vs. fairness violation.
    
    Args:
        results: Dictionary with gamma, error, and fairness_violation lists
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Sort points by error
    idx = np.argsort(results['error'])
    errors = np.array(results['error'])[idx]
    violations = np.array(results['fairness_violation'])[idx]
    gammas = np.array(results['gamma'])[idx]
    
    # Plot Pareto curve
    plt.plot(errors, violations, 'o-', color='tab:blue')
    
    # Annotate points with gamma values
    for i, (error, violation, gamma) in enumerate(zip(errors, violations, gammas)):
        plt.annotate(f'γ={gamma:.2f}', (error, violation), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    plt.xlabel('Classification Error')
    plt.ylabel('Fairness Violation')
    plt.title('Pareto Curve: Error vs. Fairness Violation')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def main(args):
    # Load training data with ID mapping
    X, y, id_to_index = load_training_data(args.data_path)
    
    # Load constraints
    constraints = load_constraints(args.constraints_path)
    
    # Compute constraint weights
    weights = compute_constraint_weights(constraints)
    
    if args.mode == 'single':
        # Run no-regret algorithm with a single gamma value
        algorithm = NoRegretFairness(
            X=X,
            y=y,
            constraint_weights=weights,
            id_to_index=id_to_index,  # Add this parameter
            gamma=args.gamma,
            eta=args.eta,
            C_lambda=args.c_lambda,
            C_tau=args.c_tau,
            time_horizon=args.iterations
        )
        
        algorithm.fit(verbose=True)
        
        # Plot convergence
        plot_convergence(
            algorithm.errors, 
            algorithm.fairness_violations, 
            args.gamma,
            os.path.join(args.output_dir, f"convergence_gamma_{args.gamma:.2f}.png") if args.output_dir else None
        )
        
    elif args.mode == 'pareto':
        # Generate Pareto curve for multiple gamma values
        gammas = np.linspace(args.min_gamma, args.max_gamma, args.num_gammas)
        print(f"Data shape: {X.shape}, Target shape: {y.shape}")
        print(f"Target class distribution: {np.bincount(y)}")
        print(f"Constraints loaded: {len(constraints)}")
        algorithm = NoRegretFairness(
            X=X,
            y=y,
            constraint_weights=weights,
            id_to_index=id_to_index,  # Add this parameter
            C_lambda=args.c_lambda,
            C_tau=args.c_tau,
            time_horizon=args.iterations
        )
        
        results = algorithm.get_pareto_curve(gammas)
        
        # Plot Pareto curve
        plot_pareto_curve(
            results,
            os.path.join(args.output_dir, "pareto_curve.png") if args.output_dir else None
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run No-Regret Fairness Algorithm')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (parquet file)')
    parser.add_argument('--constraints_path', type=str, required=True,
                        help='Path to constraints JSON file')
    parser.add_argument('--mode', type=str, choices=['single', 'pareto'], default='single',
                        help='Mode: single run or Pareto curve generation')
    
    # General parameters
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations for no-regret algorithm')
    parser.add_argument('--c_lambda', type=float, default=10.0,
                        help='C_lambda parameter')
    parser.add_argument('--c_tau', type=float, default=10.0,
                        help='C_tau parameter')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output plots')
    
    # Parameters for single mode
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma parameter (used in single mode)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='Eta parameter (used in single mode)')
    
    # Parameters for Pareto mode
    parser.add_argument('--min_gamma', type=float, default=0.0,
                        help='Minimum gamma value for Pareto curve')
    parser.add_argument('--max_gamma', type=float, default=1.0,
                        help='Maximum gamma value for Pareto curve')
    parser.add_argument('--num_gammas', type=int, default=10,
                        help='Number of gamma values for Pareto curve')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)