
import pickle
import matplotlib.pyplot as plt

# Load the single results file
with open("results_gamma_0.3.pkl", "rb") as f:
    all_results = pickle.load(f)

# Custom color palette
colors = ['#882255', '#AA4499', '#CC6677', '#DDCC77', 
'#88CCEE', '#44AA99', '#117733', '#332288', '#A9A9A9', '#000000']
# Create plot
plt.figure(figsize=(12, 8))
plt.title("Trajectory for Judge with Multiple γ Values")
plt.xlabel("Classification Error")
plt.ylabel("Maximum Fairness Violation")

# Plot each gamma trajectory
for i, gamma in enumerate(sorted(all_results.keys())):
    result = all_results[gamma]
    errors = result['errors']
    violations = result['max_violations']
    
    color = colors[i % len(colors)]
    plt.plot(errors, violations, color=color, label=f"γ={gamma}", linewidth=2)
    plt.scatter(errors[0], violations[0], color=color, s=80, marker='o', edgecolor='black')
    plt.scatter(errors[-1], violations[-1], color=color, s=80, marker='x', linewidth=2)

# Plot Pareto frontier
final_errors = [all_results[g]['final_error'] for g in sorted(all_results.keys())]
final_violations = [all_results[g]['final_max_violation'] for g in sorted(all_results.keys())]
plt.plot(final_errors, final_violations, 'k--', alpha=0.7, label="Pareto frontier")

plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("multi_gamma_trajectory.png", dpi=300)
plt.show()