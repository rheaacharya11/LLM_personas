import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

# === CONFIG ===
GAMMA_EVAL = 0.05  # you can change this easily
CATEGORICAL_COLUMNS = ['sex', 'race', 'c_charge_degree']

# === LOAD FILES ===
with open("multi_persona_data/persona_metadata.json") as f:
    persona_metadata = json.load(f)

with open("results_gamma_0.15.pkl", "rb") as f:
    results = pickle.load(f)
final_model = results[0.15]['final_model']

with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("encoded_feature_names.pkl", "rb") as f:
    encoded_feature_names = pickle.load(f)
with open("train_feature_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)
with open("multi_persona_data/final_holdout.json", "r") as f:
    test_constraints_raw = json.load(f)

# === LOAD TEST DATA ===
test_df = pd.read_parquet("test1000_subset.parquet").tail(800)
y_test = test_df['two_year_recid'].values
ids_test = test_df['id'].tolist()

# === PREPROCESS TEST DATA ===
categorical_data = test_df[CATEGORICAL_COLUMNS].fillna('Unknown')
encoded_test = encoder.transform(categorical_data)

test_df_proc = test_df.drop(columns=CATEGORICAL_COLUMNS + ['two_year_recid'])
encoded_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=test_df_proc.index)
test_df_proc = pd.concat([test_df_proc, encoded_df], axis=1)

# Align with training columns
for col in train_columns:
    if col not in test_df_proc.columns:
        test_df_proc[col] = 0
test_df_proc = test_df_proc[train_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

X_test = test_df_proc.values

# === PREDICT & EVALUATE ===
y_pred = final_model.predict(X_test)
probs_test = final_model.predict_proba(X_test)[:, 1]
probs_by_id = dict(zip(ids_test, probs_test))

test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# === FAIRNESS LOSS EVALUATION ===
def compute_persona_fairness_losses(probs_by_id, constraints_raw, gamma):
    losses = {}
    for pid, constraints in constraints_raw.items():
        violations = []
        for c in constraints:
            i, j = c["pair"]
            if i in probs_by_id and j in probs_by_id:
                v = max(0, probs_by_id[i] - probs_by_id[j] - gamma)
                violations.append(v)
        if violations:
            losses[pid] = sum(violations) / len(violations)
    return losses

def summarize_by_group(losses, metadata, attribute, agg='mean'):
    group_stats = {}
    group_map = {}
    for pid, loss in losses.items():
        group = metadata.get(pid, {}).get(attribute)
        if group:
            group_map.setdefault(group, []).append(loss)

    for group, loss_list in group_map.items():
        if agg == 'mean':
            group_stats[group] = {
                'mean': np.mean(loss_list),
                'std': np.std(loss_list),
                'count': len(loss_list)
            }
        elif agg == 'max':
            group_stats[group] = max(loss_list)

    return group_stats

# === EVALUATE FOR SELECTED GAMMA ===
persona_losses = compute_persona_fairness_losses(probs_by_id, test_constraints_raw, GAMMA_EVAL)

# Summary stats
loss_vals = list(persona_losses.values())
print(f"\n🎯 Mean Persona Fairness Loss (γ={GAMMA_EVAL}): {np.mean(loss_vals):.4f}")
print(f"📉 Min: {np.min(loss_vals):.4f}, 📈 Max: {np.max(loss_vals):.4f}, 📊 Std Dev: {np.std(loss_vals):.4f}")

# === PLOT HISTOGRAM OF ALL PERSONA LOSSES ===
plt.figure(figsize=(8, 6))
plt.hist(loss_vals, bins=20, color='mediumseagreen', edgecolor='black')
plt.title(f"Distribution of Persona Fairness Losses (γ={GAMMA_EVAL})")
plt.xlabel("Fairness Loss per Persona")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"hist_fairness_gamma_{GAMMA_EVAL:.2f}.png", dpi=300)
plt.show()

# === FAIRNESS LOSS STATS BY RACE (ALL DATA) ===
race_mean_stats = summarize_by_group(persona_losses, persona_metadata, "race", agg='mean')
print(f"\n📊 Fairness Loss by Race (γ = {GAMMA_EVAL:.2f}):")
for race, stats in race_mean_stats.items():
    print(f"{race:30s} | Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | N = {stats['count']}")

# Bar plot for mean fairness loss by race (all data)
races = list(race_mean_stats.keys())
means = [race_mean_stats[r]['mean'] for r in races]
errors = [race_mean_stats[r]['std'] for r in races]

plt.figure(figsize=(10, 6))
plt.bar(races, means, yerr=errors, capsize=5, color='coral', edgecolor='black')
plt.title(f"Mean Fairness Loss by Race (γ={GAMMA_EVAL})")
plt.ylabel("Mean Fairness Loss")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"bar_mean_fairness_by_race_gamma_{GAMMA_EVAL:.2f}.png", dpi=300)
plt.show()

# Bar plot for maximum fairness loss by race (all data)
race_max_stats = summarize_by_group(persona_losses, persona_metadata, "race", agg='max')
print(f"\n📊 Maximum Fairness Loss by Race (γ = {GAMMA_EVAL:.2f}):")
for race, max_loss in race_max_stats.items():
    print(f"{race:30s} | Max Loss: {max_loss:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(races, [race_max_stats[r] for r in races], color='coral', edgecolor='black')
plt.title(f"Maximum Fairness Loss by Race (γ={GAMMA_EVAL})")
plt.ylabel("Max Fairness Loss")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"bar_max_fairness_by_race_gamma_{GAMMA_EVAL:.2f}.png", dpi=300)
plt.show()

# === RANDOM SAMPLING AND SUMMARY FOR SELECTED RACES (white, asian, black) ===
def sample_summarize_by_group(losses, metadata, attribute, sample_size=50):
    """
    For each group (filtered to only white, asian, black), randomly sample up to `sample_size`
    losses and compute summary statistics (mean, max, std, count).
    """
    group_stats = {}
    group_map = {}
    # Build a mapping from group (race) to all associated losses
    for pid, loss in losses.items():
        group = metadata.get(pid, {}).get(attribute)
        if group in ['White', 'Asian', 'Black or African American']:
            group_map.setdefault(group, []).append(loss)
    
    # For each group, randomly sample and compute statistics
    for group, loss_list in group_map.items():
        if len(loss_list) > sample_size:
            sampled_losses = random.sample(loss_list, sample_size)
        else:
            sampled_losses = loss_list  # Use all if fewer than sample_size
        group_stats[group] = {
            'mean': np.mean(sampled_losses),
            'max': np.max(sampled_losses),
            'std': np.std(sampled_losses),
            'count': len(sampled_losses)
        }
    return group_stats

# Compute sampled statistics for the races white, asian, and black
race_sampled_stats = sample_summarize_by_group(persona_losses, persona_metadata, "race", sample_size=50)

print(f"\n📊 Sampled Fairness Loss by Race (50 personas per group if available, γ = {GAMMA_EVAL:.2f}):")
for race, stats in race_sampled_stats.items():
    print(f"{race:10s} | Mean: {stats['mean']:.4f} | Max: {stats['max']:.4f} | Std: {stats['std']:.4f} | N = {stats['count']}")

# Bar plot for sampled mean fairness loss by race
races_sampled = list(race_sampled_stats.keys())
sampled_means = [race_sampled_stats[r]['mean'] for r in races_sampled]
sampled_std = [race_sampled_stats[r]['std'] for r in races_sampled]

plt.figure(figsize=(10, 6))
plt.bar(races_sampled, sampled_means, yerr=sampled_std, capsize=5, color='coral', edgecolor='black')
plt.title(f"Mean Fairness Loss by Race (Sampled 50 per group, γ={GAMMA_EVAL})")
plt.ylabel("Mean Fairness Loss")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"bar_sampled_mean_fairness_by_race_gamma_{GAMMA_EVAL:.2f}.png", dpi=300)
plt.show()

# Bar plot for sampled maximum fairness loss by race
sampled_max = [race_sampled_stats[r]['max'] for r in races_sampled]

plt.figure(figsize=(10, 6))
plt.bar(races_sampled, sampled_max, color='coral', edgecolor='black')
plt.title(f"Maximum Fairness Loss by Race (Sampled 50 per group, γ={GAMMA_EVAL})")
plt.ylabel("Maximum Fairness Loss")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"bar_sampled_max_fairness_by_race_gamma_{GAMMA_EVAL:.2f}.png", dpi=300)
plt.show()
