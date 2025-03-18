import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.decomposition import PCA
import seaborn as sns
import os
import time
from datetime import datetime

# Create results directory if it doesn't exist
os.makedirs("../results/representation_bounds", exist_ok=True)

def load_and_preprocess_judgments(csv_path):
    """
    Load and preprocess the judgments from the CSV file
    Returns a matrix where rows=personas, columns=comparisons, values=judgments (encoded)
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract unique personas and comparisons
    personas = df['persona_id'].unique()
    comparisons = df['comparison_id'].unique()
    
    # Create a mapping for choices
    choice_mapping = {
        'similar': 0,
        'x_higher_than_y': 1,
        'y_higher_than_x': 2,
        'different': 3  # Adding this in case it appears in your data
    }
    
    # Create an empty matrix to hold judgments
    judgments = np.zeros((len(personas), len(comparisons)), dtype=int)
    
    # Fill the matrix with encoded judgments
    for _, row in df.iterrows():
        p_idx = np.where(personas == row['persona_id'])[0][0]
        c_idx = np.where(comparisons == row['comparison_id'])[0][0]
        choice = row['choice_type']
        
        # Handle potential missing choice_type values
        if pd.notna(choice) and choice in choice_mapping:
            judgments[p_idx, c_idx] = choice_mapping[choice]
        else:
            # Use -1 to indicate missing/invalid
            judgments[p_idx, c_idx] = -1
    
    return judgments, personas, comparisons

def calculate_distribution(judgments):
    """Calculate the distribution of judgments for each comparison"""
    n_comparisons = judgments.shape[1]
    n_classes = 4  # Adjust based on your data - includes 'different' option
    
    distributions = np.zeros((n_comparisons, n_classes))
    
    for i in range(n_comparisons):
        # Count occurrences of each judgment type for this comparison
        valid_judgments = judgments[:, i][judgments[:, i] >= 0]  # Filter out -1 values
        if len(valid_judgments) > 0:
            counts = np.bincount(valid_judgments, minlength=n_classes)
            # Normalize to get probabilities
            distributions[i] = counts / np.sum(counts)
    
    return distributions

def total_variation_distance(dist1, dist2):
    """Calculate total variation distance between two probability distributions"""
    return 0.5 * np.sum(np.abs(dist1 - dist2), axis=1).mean()

def estimate_vc_dimension(judgments):
    """Estimate the VC dimension using PCA on judgment patterns"""
    # Use all personas for the most accurate estimate
    # Convert judgments to float type for PCA
    judgments_float = judgments.astype(float)
    
    # Fill -1 values (missing) with the mean of that comparison
    for j in range(judgments_float.shape[1]):
        valid_idx = judgments_float[:, j] >= 0
        if np.any(valid_idx):
            mean_val = np.mean(judgments_float[valid_idx, j])
            judgments_float[~valid_idx, j] = mean_val
    
    # Run PCA
    pca = PCA()
    pca.fit(judgments_float)
    
    # Count components needed to explain 90% variance
    var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(var_ratio)
    n_components = np.argmax(cumulative_var >= 0.9) + 1
    
    print(f"PCA analysis: {n_components} components explain 90% of variance")
    
    # For visualization/debugging, plot the explained variance
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(var_ratio)+1), cumulative_var, 'o-')
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.axvline(x=n_components, color='g', linestyle='--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig('figures/pca_variance_explained.png')
    plt.close()
    
    # Scale components to get VC dimension
    # Generally, VC dimension is related to the intrinsic dimensionality
    vc_dim = max(1, int(n_components * 1.5))
    
    return vc_dim

def run_single_trial(all_judgments, k, full_dist):
    """Run a single trial for a specific k value"""
    # Sample k personas randomly
    indices = random.sample(range(all_judgments.shape[0]), k)
    sampled_judgments = all_judgments[indices]
    
    # Calculate empirical distribution from the sample
    sample_dist = calculate_distribution(sampled_judgments)
    
    # Calculate error (total variation distance)
    return total_variation_distance(sample_dist, full_dist)

def verify_representation_bound(all_judgments, n_trials=20, confidence=0.05):
    """
    Calculate empirical errors and theoretical bounds for different sample sizes
    
    Args:
        all_judgments: Matrix of all judgment data (personas x comparisons)
        n_trials: Number of random samples to try for each k
        confidence: Confidence level (e.g., 0.05 for 95% confidence)
        
    Returns:
        k_values, avg_errors, std_errors, theoretical_bounds
    """
    n_personas, n_pairs = all_judgments.shape
    
    # Calculate the full distribution (all personas)
    full_dist = calculate_distribution(all_judgments)
    
    # Estimate VC dimension
    d = estimate_vc_dimension(all_judgments)
    print(f"Estimated VC dimension: {d}")
    
    # Test different k values (number of personas to sample)
    # Use smaller range for demonstration in the notebook
    k_values = [5, 10, 20, 50, 100, 200, 500]
    k_values = [k for k in k_values if k < n_personas]
    
    avg_errors = []
    std_errors = []
    theoretical_bounds = []
    
    for i, k in enumerate(k_values):
        print(f"Processing k={k} ({i+1}/{len(k_values)})...")
        
        # Run trials sequentially
        errors = []
        for t in range(n_trials):
            errors.append(run_single_trial(all_judgments, k, full_dist))
        
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        avg_errors.append(avg_error)
        std_errors.append(std_error)
        
        # Calculate theoretical bound (Hoeffding's inequality with VC dimension)
        c = 2.0  # This constant can be calibrated
        bound = c * np.sqrt((d * np.log(k) + np.log(1/confidence)) / k)
        theoretical_bounds.append(bound)
    
    return k_values, avg_errors, std_errors, theoretical_bounds, d

# VISUALIZATION 1: Enhanced Original Plot with Reference Lines
def plot_with_reference_lines(k_values, avg_errors, std_errors, theoretical_bounds, params):
    """Plot with reference lines at key error thresholds"""
    plt.figure(figsize=(12, 8))
    
    # Plot empirical error with error bars
    plt.errorbar(k_values, avg_errors, yerr=std_errors, fmt='bo-', capsize=5, 
                label='Empirical Error (mean ± std)')
    
    # Plot theoretical bound
    plt.plot(k_values, theoretical_bounds, 'r--', linewidth=2, label='Theoretical Bound')
    
    # Add fitted curve
    x_smooth = np.logspace(np.log10(min(k_values)), np.log10(max(k_values)*1.2), 100)
    y_smooth = params[0] * x_smooth**(-params[1])
    plt.plot(x_smooth, y_smooth, 'g-', linewidth=2, 
             label=f'Fitted curve: {params[0]:.2f}·k^(-{params[1]:.2f})')
    
    # Add reference lines for key error thresholds
    thresholds = [0.1, 0.05, 0.02, 0.01]
    colors = ['purple', 'orange', 'brown', 'pink']
    
    for threshold, color in zip(thresholds, colors):
        # Calculate k needed for this threshold
        k_needed = int(np.ceil((params[0] / threshold) ** (1/params[1])))
        
        # Add horizontal line at threshold
        plt.axhline(y=threshold, color=color, linestyle=':', alpha=0.7)
        
        # Add vertical line at k needed if within range
        if k_needed <= max(k_values) * 1.2:
            plt.axvline(x=k_needed, color=color, linestyle=':', alpha=0.7)
            
            # Add annotation
            plt.annotate(f'k ≈ {k_needed}', 
                        xy=(k_needed, threshold), 
                        xytext=(k_needed * 1.1, threshold * 1.1),
                        color=color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Personas (k)', fontsize=14)
    plt.ylabel('Total Variation Distance', fontsize=14)
    plt.title('Representation Bound with Error Thresholds', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/bound_with_thresholds.png', dpi=300)
    plt.show()

# VISUALIZATION 2: Diminishing Returns Analysis    
def plot_diminishing_returns(k_values, avg_errors, params):
    """Visualize diminishing returns in error reduction as personas increase"""
    plt.figure(figsize=(12, 8))
    
    # Plot empirical error
    plt.subplot(2, 1, 1)
    plt.plot(k_values, avg_errors, 'bo-', linewidth=2, label='Empirical Error')
    
    # Plot fitted curve
    x_smooth = np.logspace(np.log10(min(k_values)), np.log10(max(k_values)*1.2), 100)
    y_smooth = params[0] * x_smooth**(-params[1])
    plt.plot(x_smooth, y_smooth, 'g-', linewidth=2, 
             label=f'Fitted curve: {params[0]:.2f}·k^(-{params[1]:.2f})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Total Variation Distance', fontsize=14)
    plt.title('Error vs Number of Personas', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    # Plot improvement per additional persona
    plt.subplot(2, 1, 2)
    
    # Calculate error reduction per additional persona
    improvement = []
    for i in range(len(k_values)-1):
        personas_added = k_values[i+1] - k_values[i]
        error_reduced = avg_errors[i] - avg_errors[i+1]
        improvement.append(error_reduced / personas_added)
    
    # Plot improvement
    plt.bar(k_values[:-1], improvement, width=0.4*np.array(k_values[:-1]), alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Number of Personas (k)', fontsize=14)
    plt.ylabel('Error Reduction per Added Persona', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figures/diminishing_returns.png', dpi=300)
    plt.show()

# VISUALIZATION 3: Bar Chart of Personas Needed
def plot_personas_needed(params):
    """Create a bar chart showing personas needed for different error thresholds"""
    plt.figure(figsize=(10, 6))
    
    # Define thresholds
    thresholds = [0.15, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    # Calculate personas needed
    personas_needed = [int(np.ceil((params[0] / threshold) ** (1/params[1]))) for threshold in thresholds]
    
    # Create bar chart
    bars = plt.bar(range(len(thresholds)), personas_needed, width=0.7, alpha=0.8)
    
    # Customize appearance
    plt.xticks(range(len(thresholds)), [f'≤ {t:.3f}' for t in thresholds])
    plt.xlabel('Error Threshold', fontsize=14)
    plt.ylabel('Number of Personas Needed', fontsize=14)
    plt.title('Personas Required for Different Error Thresholds', fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/personas_needed.png', dpi=300)
    plt.show()

# VISUALIZATION 4: Specific Comparison Examples
def plot_specific_comparisons(all_judgments, personas, comparisons, k_values):
    """Visualize how distributions converge for specific comparison examples"""
    # Calculate full distribution
    full_dist = calculate_distribution(all_judgments)
    
    # Find most polarized comparisons (closest to uniform distribution)
    polarization = np.sum(np.abs(full_dist - 1/3), axis=1)
    valid_idx = ~np.isnan(polarization)  # Filter out NaN values
    most_polarized_idx = np.argmin(polarization[valid_idx])
    
    # Find most unanimous comparison
    unanimity = np.max(full_dist, axis=1)
    valid_idx = ~np.isnan(unanimity)  # Filter out NaN values
    most_unanimous_idx = np.argmax(unanimity[valid_idx])
    
    # Comparison indices to visualize
    comp_indices = [most_polarized_idx, most_unanimous_idx]
    comp_titles = ["Most Polarized Comparison", "Most Unanimous Comparison"]
    
    plt.figure(figsize=(15, 10))
    
    for i, (comp_idx, title) in enumerate(zip(comp_indices, comp_titles)):
        # Plot for different sample sizes
        for j, k in enumerate([5, 50, 200, 500]):
            if k >= len(all_judgments):
                continue  # Skip if k is larger than available personas
                
            plt.subplot(2, 4, i*4 + j + 1)
            
            # Sample k personas 10 times and average
            sample_dists = []
            for _ in range(10):
                indices = random.sample(range(all_judgments.shape[0]), k)
                sampled_judgments = all_judgments[indices]
                sample_dist = calculate_distribution(sampled_judgments)
                sample_dists.append(sample_dist[comp_idx])
            
            avg_sample_dist = np.mean(sample_dists, axis=0)
            
            # Plot distribution
            labels = ['Similar', 'X > Y', 'Y > X', 'Different']
            plt.bar(labels, avg_sample_dist, alpha=0.7)
            plt.bar(labels, full_dist[comp_idx], alpha=0.3, color='red')
            
            # Calculate TVD for this comparison
            tvd = 0.5 * np.sum(np.abs(avg_sample_dist - full_dist[comp_idx]))
            
            plt.title(f'k={k}, TVD={tvd:.3f}')
            if j == 0:  # Add y-label only for leftmost plots
                plt.ylabel(title)
            if i == 1:  # Add x-label only for bottom plots
                plt.xlabel('Judgment')
            
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figures/specific_comparisons.png', dpi=300)
    plt.show()

# VISUALIZATION 5: Dual-Axis Plot
def plot_dual_axis(k_values, avg_errors, theoretical_bounds, params):
    """Create a dual-axis plot to better show empirical and theoretical bounds"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # First axis for empirical error
    ax1.set_xlabel('Number of Personas (k)', fontsize=14)
    ax1.set_ylabel('Empirical Error', fontsize=14, color='blue')
    ax1.plot(k_values, avg_errors, 'bo-', linewidth=2, label='Empirical Error')
    
    # Add fitted curve
    x_smooth = np.logspace(np.log10(min(k_values)), np.log10(max(k_values)*1.2), 100)
    y_smooth = params[0] * x_smooth**(-params[1])
    ax1.plot(x_smooth, y_smooth, 'g-', linewidth=2, 
             label=f'Fitted curve: {params[0]:.2f}·k^(-{params[1]:.2f})')
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Second axis for theoretical bound
    ax2 = ax1.twinx()
    ax2.set_ylabel('Theoretical Bound', fontsize=14, color='red')
    ax2.plot(k_values, theoretical_bounds, 'r--', linewidth=2, label='Theoretical Bound')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')
    
    # Add a title
    plt.title('Representation Bounds with Dual Axes', fontsize=16)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/dual_axis.png', dpi=300)
    plt.show()

# Main execution function
def analyze_representation_bounds(csv_path, n_trials=20):
    """Run the complete representation bounds analysis"""
    start_time = time.time()
    print(f"Starting analysis at {datetime.now()}")
    
    # Load and process the data
    print(f"Loading data from {csv_path}")
    all_judgments, personas, comparisons = load_and_preprocess_judgments(csv_path)
    print(f"Loaded judgments matrix: {all_judgments.shape}")
    
    # Calculate representation bounds
    print("Calculating representation bounds...")
    k_values, avg_errors, std_errors, bounds, vc_dim = verify_representation_bound(
        all_judgments, n_trials=n_trials, confidence=0.05)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'k_values': k_values,
        'avg_errors': avg_errors,
        'std_errors': std_errors,
        'theoretical_bounds': bounds
    })
    results_df.to_csv("../results/representation_bounds/bounds_data.csv", index=False)
    
    # Fit power law to the data
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b):
        return a * x**(-b)
    
    params, _ = curve_fit(power_law, np.array(k_values), np.array(avg_errors))
    print(f"Fitted power law parameters: a={params[0]:.4f}, b={params[1]:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Enhanced Original Plot with Reference Lines
    plot_with_reference_lines(k_values, avg_errors, std_errors, bounds, params)
    
    # 2. Diminishing Returns Analysis  
    plot_diminishing_returns(k_values, avg_errors, params)
    
    # 3. Bar Chart of Personas Needed
    plot_personas_needed(params)
    
    # 4. Specific Comparison Examples
    plot_specific_comparisons(all_judgments, personas, comparisons, k_values)
    
    # 5. Dual-Axis Plot
    plot_dual_axis(k_values, avg_errors, bounds, params)
    
    # Calculate required personas for various thresholds
    thresholds = [0.1, 0.05, 0.02, 0.01]
    print("\nNumber of personas needed to achieve error thresholds:")
    for threshold in thresholds:
        required_k = int(np.ceil((params[0] / threshold) ** (1/params[1])))
        print(f"Error ≤ {threshold:.2f}: Approximately {required_k} personas")
    
    # Report timing
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    return results_df, params, vc_dim

# Example usage with full path to the CSV file 
# (You'll need to update this path to your combined_results.csv)
if __name__ == "__main__":
    csv_path = "../results/persona_size/combined_results.csv"
    if os.path.exists(csv_path):
        results_df, params, vc_dim = analyze_representation_bounds(csv_path, n_trials=20)
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please update the path to your combined_results.csv file")