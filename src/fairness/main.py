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

def debug_constraints_and_data(constraints_path, data_path, test_data_path):
    """
    Comprehensive debug function to inspect constraints and data
    """
    # Load data
    try:
        if test_data_path:
            X_train, y_train, X_test, y_test, id_to_index = load_training_data(data_path, test_data_path)
        else:
            X_train, y_train, id_to_index = load_training_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load constraints
    try:
        constraints = load_constraints(args.constraints_path)
    except Exception as e:
        print(f"Error loading constraints: {e}")
        return

    # Detailed constraint analysis
    print("\n--- Data and Constraint Analysis ---")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Training label distribution: {np.bincount(y_train)}")
    print(f"Total constraints loaded: {len(constraints)}")

    # Examine constraint indices
    valid_constraints = {}
    invalid_indices = 0
    conflicting_labels = 0

    should_print = 0
    for (i, j), judges in list(constraints.items()):
        if i >= len(y_train) or j >= len(y_train):
            invalid_indices += 1
            continue
        
        if y_train[i] != y_train[j]:
            conflicting_labels += 1
            if should_print < 5:
                print(i, j)
                should_print += 1
        
        valid_constraints[(i,j)] = judges

    print(f"\nConstraint Analysis:")
    print(f"  Total constraints: {len(constraints)}")
    print(f"  Constraints with out-of-bounds indices: {invalid_indices}")
    print(f"  Constraints with conflicting labels: {conflicting_labels}")
    print(f"  Remaining valid constraints: {len(valid_constraints)}")

    # Sample constraint details
    print("\nSample Constraints:")
    for (i, j), judges in list(valid_constraints.items())[:10]:
        print(f"  Pair ({i}, {j}): Label1 = {y_train[i]}, Label2 = {y_train[j]}, Judges = {len(judges)}")


def main(args):
    
    data_result = load_training_data(args.data_path, args.test_data_path)

    
    if args.test_data_path:
        X_train, y_train, X_test, y_test, id_to_index = data_result
    else:
        X_train, y_train, id_to_index = data_result
        X_test, y_test = None, None

    
    # Load constraints
    constraints = load_constraints(args.constraints_path)
    
    # Compute constraint weights
    weights = compute_constraint_weights(constraints)
    print(f"Computed constraint weights")

    
    
    if args.mode == 'single':
        # Run no-regret algorithm with a single gamma value
        algorithm = NoRegretFairness(
            X=X_train,
            y=y_train,
            constraint_weights=weights,
            id_to_index=id_to_index,
            X_test=X_test,  # Pass test data
            y_test=y_test,   # Pass test data
            gamma=args.gamma,
            eta=args.eta,
            C_lambda=args.c_lambda,
            C_tau=args.c_tau,
            time_horizon=args.iterations
        )
        
        algorithm.fit(verbose=True)
        test_results = algorithm.evaluate_on_test(X_test, y_test)
        print(f"\nFinal evaluation on test data:")
        print(f"  Test error: {test_results['test_error']:.4f}")

        print(f"  Test accuracy: {test_results['test_accuracy']:.4f}")
        
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
def test_cost_computation():
    # Create minimal test dataset
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    y_test = np.array([0, 1, 0])
    
    # Create a constraint between sample 0 and 1
    constraint_weights = {(0, 1): 0.5}
    
    # Initialize algorithm with test data
    alg = NoRegretFairness(X_test, y_test, constraint_weights, gamma=0.1)
    
    # Set some lambda values
    lambda_vals = {(0, 1): 0.2}
    
    # Compute costs
    costs = alg.compute_costs(lambda_vals)
    
    # Expected costs:
    # Sample 0: y=0, cost_0=0, cost_1=1/3 + 0.2 (lambda) = 0.533
    # Sample 1: y=1, cost_0=1/3 + 0 (lambda) = 0.333, cost_1=0
    # Sample 2: y=0, cost_0=0, cost_1=1/3 = 0.333
    
    print("Test Data Costs:")
    for i, (cost_0, cost_1) in enumerate(costs):
        print(f"Sample {i}: cost_0={cost_0:.6f}, cost_1={cost_1:.6f}")
    
    # Verify results
    expected = [(0.0, 0.533333), (0.333333, 0.0), (0.0, 0.333333)]
    for i in range(len(costs)):
        assert abs(costs[i][0] - expected[i][0]) < 1e-5, f"Cost_0 for sample {i} doesn't match"
        assert abs(costs[i][1] - expected[i][1]) < 1e-5, f"Cost_1 for sample {i} doesn't match"
    
    print("Test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run No-Regret Fairness Algorithm')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (parquet file)')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to testing data (parquet file)')
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
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='Gamma parameter (used in single mode)')
    parser.add_argument('--eta', type=float, default=0.1,
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
    #test_cost_computation()
    main(args)