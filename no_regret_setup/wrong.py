import json
import pickle
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from no_regret import FairnessElicitationAlgorithm  # Your class here!

TOL_COLORS = [
    "#117733",  # green
    "#332288",  # navy
    "#44AA99",  # teal
    "#88CCEE",  # light blue
    "#DDCC77",  # sand
    "#CC6677",  # red-pink
    "#AA4499",  # purple
    "#882255",  # wine
    "#661100",  # brown
    "#999933",  # olive
]

# --- Load persona metadata ---
with open("multi_persona_data/persona_metadata.json", "r") as f:
    persona_metadata = json.load(f)

# --- Load training constraints ---
with open("multi_persona_data/final_train.json", "r") as f:
    train_constraints_raw = json.load(f)

# --- Load test constraints ---
with open("multi_persona_data/final_holdout.json", "r") as f:
    test_constraints_raw = json.load(f)

# --- Load test data ---
test_df = pd.read_parquet("test1000_subset.parquet")
y_test = test_df["two_year_recid"].values
ids_test = test_df["id"].tolist()

with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("encoded_feature_names.pkl", "rb") as f:
    encoded_feature_names = pickle.load(f)

categorical_columns = ['sex', 'race', 'c_charge_degree']
categorical_data = test_df[categorical_columns].fillna("Unknown")
encoded_test = encoder.transform(categorical_data)
test_df_proc = test_df.drop(columns=categorical_columns + ["two_year_recid"])
encoded_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=test_df_proc.index)
test_df_proc = pd.concat([test_df_proc, encoded_df], axis=1)
X_test = test_df_proc.apply(pd.to_numeric, errors="coerce").fillna(0).values

# --- Helper: compute fairness loss on held-out test constraints ---
def compute_fairness_violation(probs_by_id, constraints_by_persona, gamma=0.1):
    all_violations = []
    for constraints in constraints_by_persona.values():
        for c in constraints:
            i, j = c["pair"]
            if i not in probs_by_id or j not in probs_by_id:
                continue
            p_i = probs_by_id[i]
            p_j = probs_by_id[j]
            violation = max(0, p_i - p_j - gamma)
            all_violations.append(violation)
    return np.mean(all_violations), np.max(all_violations)

# --- Aggregate all test constraints into one set ---
test_constraints_combined = []
for constraint_list in test_constraints_raw.values():
    test_constraints_combined.extend(constraint_list)
A_test = len(test_constraints_combined)

# --- Training on all personas ---
combined_constraints = []
for pid in train_constraints_raw:
    combined_constraints.extend(train_constraints_raw[pid])

# --- Sweep over C values ---
C_values = [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]
results = {}
gamma_eval = 0.1

for C_val in C_values:
    print(f"\n🚀 Training with C = {C_val}")

    algo = FairnessElicitationAlgorithm(
        data_path="train200_subset.parquet",
        constraint_sets_path="multi_persona_data/final_train.json",  # just for loading
        time_horizon=300,
        C_lambda=10.0,
        C_tau=1.0
    )

    algo.load_constraint_sets(custom_constraints=combined_constraints)

    # Override cost-sensitive oracle to pass in C
    algo.cost_sensitive_oracle = partial(algo.cost_sensitive_oracle, C=C_val)

    result = algo.run(gamma_values=[gamma_eval])
    model = result[gamma_eval]["final_model"]
    coefs = model.coef_.flatten()
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    for i in top_idx:
        print(f"{i}: coef = {coefs[i]:.4f}")
    # --- Evaluate on test set ---
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    probs = model.predict_proba(X_test)[:, 1]
    probs_by_id = dict(zip(ids_test, probs))

    fair_loss, max_violation = compute_fairness_violation(probs_by_id, test_constraints_raw, gamma=gamma_eval)

    results[C_val] = {
        "model": model,
        "test_acc": test_acc,
        "fair_loss": fair_loss,
        "max_violation": max_violation,
        "probs": probs
    }

# --- Analyze prediction spread and coefficient magnitude ---
stds = []
means = []
coef_means = []
coef_maxs = []

for C in C_values:
    probs = results[C]["probs"]
    stds.append(np.std(probs))
    means.append(np.mean(probs))

    coef = results[C]["model"].coef_.flatten()
    coef_means.append(np.mean(np.abs(coef)))
    coef_maxs.append(np.max(np.abs(coef)))

# --- Plot Accuracy vs C ---
plt.figure(figsize=(8, 6))
plt.plot(C_values, [results[C]['test_acc'] for C in C_values], marker='o')
plt.xscale("log")
plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs C")
plt.grid(True)
plt.savefig("wrong_script_accuracy_vs_c.png")
plt.show()

# --- Plot Fairness Loss vs C ---
plt.figure(figsize=(8, 6))
plt.plot(C_values, [results[C]['fair_loss'] for C in C_values], marker='s', color='teal')
plt.xscale("log")
plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Fairness Loss (γ = 0.1)")
plt.title("Fairness Loss vs C")
plt.grid(True)
plt.savefig("wrong_script_fairness_vs_c.png")
plt.show()

# --- Plot Prediction Distribution for each C ---
plt.figure(figsize=(10, 6))
for C in C_values:
    plt.hist(results[C]['probs'], bins=30, alpha=0.5, label=f"C={C}")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Prediction Distributions by C")
plt.legend()
plt.grid(True)
plt.savefig("wrong_script_prediction_dists_by_c.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(C_values, stds, marker='o', color='darkblue')
plt.xscale("log")
plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Std Dev of Predicted Probabilities")
plt.title("Prediction Spread vs. Regularization Strength")
plt.grid(True)
plt.savefig("std_dev_vs_C.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(C_values, coef_means, marker='s', label="Mean |coef|", color=TOL_COLORS[0])  # green
plt.plot(C_values, coef_maxs, marker='^', label="Max |coef|", color=TOL_COLORS[1])    # navy

plt.xscale("log")
plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Coefficient Magnitude")
plt.title("Model Coefficient Magnitudes vs. C")
plt.legend()
plt.grid(True)
plt.savefig("coef_magnitude_vs_C.png", dpi=300)
plt.show()


