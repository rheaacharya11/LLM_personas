import pandas as pd

# Load dataset
df = pd.read_csv("results/persona_size/combined_results.csv")
# Convert option_order from string to list
df["option_order"] = df["option_order"].apply(lambda x: eval(x))

# Map chosen_option (1, 2, 3) to its position (0, 1, 2) in option_order
df["chosen_index"] = df.apply(lambda row: row["option_order"].index(row["chosen_option"]), axis=1)

# Count occurrences of each order
order_counts = df["option_order"].value_counts()

# Check how often each position (0, 1, 2) was chosen for each order
order_choice_counts = {}
for order in order_counts.index:
    subset = df[df["option_order"].apply(lambda x: x == order)]
    count_first = (subset["chosen_index"] == 0).sum()
    count_second = (subset["chosen_index"] == 1).sum()
    count_third = (subset["chosen_index"] == 2).sum()
    
    order_choice_counts[str(order)] = {
        "total": len(subset),
        "first_position_chosen": count_first,
        "second_position_chosen": count_second,
        "third_position_chosen": count_third
    }

# Convert to DataFrame and save results
order_choice_df = pd.DataFrame.from_dict(order_choice_counts, orient="index")
order_choice_df.to_csv("persona_size/eda/order_breakdown.csv", index=True)

# Breakdown conditioned on chosen_option (1, 2, 3)
chosen_breakdown = {}
for chosen in [1, 2, 3]:  # The actual presented choices
    subset = df[df["chosen_option"] == chosen]
    chosen_counts = subset["option_order"].value_counts()
    
    chosen_counts_df = pd.DataFrame(chosen_counts)
    chosen_counts_df.to_csv(f"persona_size/eda/breakdown_chosen_{chosen}.csv")

print("Results saved as 'order_breakdown.csv' and 'breakdown_chosen_X.csv' for each chosen option.")