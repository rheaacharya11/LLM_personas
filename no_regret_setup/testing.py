import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
with open("multi_persona_data/persona_metadata.json") as f:
    persona_metadata = json.load(f)

# --- Load trained model ---
with open("results_gamma_0.15.pkl", "rb") as f:
    results = pickle.load(f)
final_model = results[0.2]['final_model']

# --- Load test data ---
test_df = pd.read_parquet("test1000_subset.parquet")
y_test = test_df['two_year_recid'].values
ids_test = test_df['id'].tolist()

# --- Load encoder and feature names ---
with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("encoded_feature_names.pkl", "rb") as f:
    encoded_feature_names = pickle.load(f)

# --- Define categorical columns (used during training) ---
categorical_columns = ['sex', 'race', 'c_charge_degree']

# --- Encode test data ---
categorical_data = test_df[categorical_columns].fillna('Unknown')
encoded_test = encoder.transform(categorical_data)
# Drop original categorical + label columns, then add encoded features
test_df_proc = test_df.drop(columns=categorical_columns + ['two_year_recid'])
encoded_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=test_df_proc.index)
test_df_proc = pd.concat([test_df_proc, encoded_df], axis=1)
def summarize_by_group(persona_losses, persona_metadata, attribute):
    group_to_losses = {}

    for pid, loss in persona_losses.items():
        if pid not in persona_metadata:
            continue
        group = persona_metadata[pid].get(attribute)
        if group is None:
            continue
        group_to_losses.setdefault(group, []).append(loss)

    # Compute mean, std per group
    group_stats = {}
    for group, losses in group_to_losses.items():
        group_stats[group] = {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'count': len(losses),
        }

    return group_stats

# Load training-time feature columns
with open("train_feature_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

# Add missing columns (fill with 0s)
for col in train_columns:
    if col not in test_df_proc.columns:
        test_df_proc[col] = 0

# Drop extra columns
for col in list(test_df_proc.columns):
    if col not in train_columns:
        test_df_proc = test_df_proc.drop(columns=col)

# Reorder to match training
test_df_proc = test_df_proc[train_columns]

# Convert to numeric and fill NaNs
test_df_proc = test_df_proc.apply(pd.to_numeric, errors='coerce').fillna(0)

X_test = test_df_proc.values

# --- Predict & compute test accuracy ---
y_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# --- Compute prediction probabilities & map to IDs ---
probs_test = final_model.predict_proba(X_test)[:, 1]
probs_by_id = {id_: prob for id_, prob in zip(ids_test, probs_test)}

# --- Load held-out constraints grouped by persona ---
with open("multi_persona_data/final_holdout.json", "r") as f:
    test_constraints_raw = json.load(f)

# --- Define fairness loss computation per persona ---
def compute_persona_fairness_losses(probs_by_id, constraints_raw, gamma):
    gamma = 0.5
    persona_losses = {}

    for persona_id, constraints in constraints_raw.items():
        violations = []

        for c in constraints:
            i, j = c["pair"]
            if i not in probs_by_id or j not in probs_by_id:
                continue  # skip missing predictions

            violation = max(0, probs_by_id[i] - probs_by_id[j] - gamma)
            violations.append(violation)

        if violations:
            loss = sum(violations) / len(violations)
            persona_losses[persona_id] = loss

    return persona_losses

# --- Evaluate at one gamma (e.g., γ = 0.1) ---
gamma = 0.1
persona_losses = compute_persona_fairness_losses(probs_by_id, test_constraints_raw, gamma)

loss_values = list(persona_losses.values())
mean_loss = np.mean(loss_values)
std_loss = np.std(loss_values)
min_loss = np.min(loss_values)
max_loss = np.max(loss_values)

print(f"\n🎯 Mean Persona Fairness Loss (γ={gamma}): {mean_loss:.4f}")
print(f"📉 Min: {min_loss:.4f}, 📈 Max: {max_loss:.4f}, 📊 Std Dev: {std_loss:.4f}")

# --- Histogram of fairness losses per persona ---
plt.figure(figsize=(8, 6))
plt.hist(loss_values, bins=20, color='mediumseagreen', edgecolor='black')
plt.title(f"Distribution of Persona Fairness Losses (γ={gamma})")
plt.xlabel("Fairness Loss per Persona")
plt.ylabel("Count")
plt.grid(True)
plt.savefig(f"persona_fairness_loss_histogram_gamma_{gamma:.2f}.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Sweep gamma values and plot mean loss ---
gammas = np.linspace(0.0, 0.25, 50)
mean_losses = []

for gamma in gammas:
    persona_losses = compute_persona_fairness_losses(probs_by_id, test_constraints_raw, gamma)
    mean_loss = np.mean(list(persona_losses.values()))
    mean_losses.append(mean_loss)

plt.figure(figsize=(8, 6))
plt.plot(gammas, mean_losses, marker='o', linestyle='-', color='teal')
plt.title("Mean Persona Fairness Loss vs γ")
plt.xlabel("γ (Slack Threshold)")
plt.ylabel("Mean Fairness Loss")
plt.grid(True)
plt.savefig("mean_persona_fairness_vs_gamma.png", dpi=300, bbox_inches='tight')
plt.show()
race_stats = summarize_by_group(persona_losses, persona_metadata, "race")
print("\n📊 Fairness Loss by Race (γ = {:.2f}):".format(gamma))
for race, stats in race_stats.items():
    print(f"{race:20s} | Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | N = {stats['count']}")
# Plot mean fairness loss by group (e.g., race)
groups = list(race_stats.keys())
means = [race_stats[g]['mean'] for g in groups]
errors = [race_stats[g]['std'] for g in groups]

plt.figure(figsize=(10, 6))
plt.bar(groups, means, yerr=errors, capsize=5, color='coral', edgecolor='black')
plt.title(f"Mean Fairness Loss by Race (γ={gamma})")
plt.ylabel("Mean Fairness Loss")
plt.xlabel("Race")
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"fairness_by_race_gamma_{gamma:.2f}.png", dpi=300)
plt.show()

def summarize_by_group_max_loss(persona_losses, persona_metadata, attribute):
    group_to_losses = {}

    for pid, loss in persona_losses.items():
        if pid not in persona_metadata:
            continue
        group = persona_metadata[pid].get(attribute)
        if group is None:
            continue
        group_to_losses.setdefault(group, []).append(loss)

    # Compute max loss per group
    group_max_losses = {group: max(losses) for group, losses in group_to_losses.items()}

    return group_max_losses


# Calculate maximum fairness loss by race
race_max_losses = summarize_by_group_max_loss(persona_losses, persona_metadata, "race")

# Display results
print("\n📊 Maximum Fairness Loss by Race (γ = {:.2f}):".format(gamma))
for race, max_loss in race_max_losses.items():
    print(f"{race:20s} | Max Loss: {max_loss:.4f}")


import matplotlib.pyplot as plt

# Extract races and their corresponding maximum losses
groups = list(race_max_losses.keys())
max_losses = [race_max_losses[g] for g in groups]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(groups, max_losses, color='coral', edgecolor='black')
plt.title(f"Maximum Fairness Loss by Race (γ={gamma})")
plt.ylabel("Maximum Fairness Loss")
plt.xlabel("Race")
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"max_fairness_loss_by_race_gamma_{gamma:.2f}.png", dpi=300)
plt.show()