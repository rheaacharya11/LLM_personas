import json
import matplotlib.pyplot as plt

# --- Load constraint data ---
with open("multi_persona_data/final_train.json", "r") as f:
    constraint_data = json.load(f)

# --- Count selected pairs per judge ---
selected_counts = []
for judge_id, constraints in constraint_data.items():
    count = sum(1 for c in constraints if c["weight"] > 0)
    selected_counts.append(count)

# --- Compute basic stats ---
min_selected = min(selected_counts)
max_selected = max(selected_counts)
mean_selected = sum(selected_counts) / len(selected_counts)

# --- Print basic stats ---
print(f"Min selected: {min_selected}")
print(f"Max selected: {max_selected}")
print(f"Mean selected: {mean_selected:.2f}")

# --- Plot histogram ---
plt.figure(figsize=(8, 6))
plt.hist(selected_counts, bins=20, color='#88CCEE', edgecolor='grey')
plt.title("Number of Constraints Selected per Judge")
plt.xlabel("Number of Selected Pairs")
plt.ylabel("Number of Judges")
plt.grid(True)

# --- Add summary stats text box ---
summary_text = (
    f"Min: {min_selected}\n"
    f"Max: {max_selected}\n"
    f"Mean: {mean_selected:.2f}"
)
plt.text(0.95, 0.95, summary_text,
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey"))

# --- Save and show ---
plt.savefig("judge_constraint_histogram.png", dpi=300, bbox_inches='tight')
plt.show()
