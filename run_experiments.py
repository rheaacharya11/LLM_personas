import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from no_regret import FairnessElicitationAlgorithm  # Import your core class

# --- Load persona metadata ---
with open("multi_persona_data/persona_metadata.json", "r") as f:
    persona_metadata = json.load(f)

# --- Choose experiment configuration ---
EXPERIMENT_TYPE = "group"  # options: "single" or "group"
TARGET_PERSONA = "007"
TARGET_GROUP = {"race": "Asian"}  # change to {"religion": "Catholic"}, etc.

# --- Load training constraints ---
with open("multi_persona_data/final_train.json", "r") as f:
    train_constraints_raw = json.load(f)

if EXPERIMENT_TYPE == "single":
    selected_personas = [TARGET_PERSONA]
elif EXPERIMENT_TYPE == "group":
    attr, val = list(TARGET_GROUP.items())[0]
    selected_personas = [
        pid for pid, meta in persona_metadata.items()
        if meta.get(attr) == val
    ]
else:
    raise ValueError("Invalid experiment type.")

# --- Aggregate constraints ---
combined_constraints = []
for pid in selected_personas:
    combined_constraints.extend(train_constraints_raw.get(pid, []))

print(f"\n🔧 Using {len(selected_personas)} persona(s) with {len(combined_constraints)} total constraints")

# --- Train the model ---
algo = FairnessElicitationAlgorithm(
    data_path="train200_subset.parquet",
    constraint_sets_path="multi_persona_data/final_train.json",  # Still needed for loading metadata
    time_horizon=500,
    C_lambda=10.0,
    C_tau=1.0
)
algo.load_constraint_sets(custom_constraints=combined_constraints)
results = algo.run(gamma_values=[0.2])
final_model = results[0.2]["final_model"]

# --- Load test data ---
test_df = pd.read_parquet("test100_subset.parquet")
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

# --- Evaluate accuracy ---
y_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# --- Predict probabilities ---
probs_test = final_model.predict_proba(X_test)[:, 1]
probs_by_id = {id_: prob for id_, prob in zip(ids_test, probs_test)}

# --- Load test constraints ---
with open("multi_persona_data/final_holdout.json", "r") as f:
    test_constraints_raw = json.load(f)

# --- Evaluate fairness loss per persona ---
gamma = 0.05
persona_losses = {}
for pid, constraint_list in test_constraints_raw.items():
    total_violation = 0.0
    count = 0
    processed = set()
    for c in constraint_list:
        i, j = c["pair"]
        weight = c["weight"]
        pair_key = tuple(sorted((i, j)))
        if pair_key in processed or i not in probs_by_id or j not in probs_by_id:
            continue
        processed.add(pair_key)
        prob_i = probs_by_id[i]
        prob_j = probs_by_id[j]
        violation = max(0, prob_i - prob_j - gamma)
        total_violation += weight * violation
        count += 1
    if count > 0:
        persona_losses[pid] = total_violation / count

# --- Compute average and max loss ---
avg_loss = np.mean(list(persona_losses.values()))
max_loss = np.max(list(persona_losses.values()))
print(f"\n📊 Avg Fairness Loss: {avg_loss:.4f}")
print(f"📊 Max Fairness Loss: {max_loss:.4f}")

# --- (Optional) Group-based generalization analysis ---
group_attr = "race"
grouped_results = {}
for pid, loss in persona_losses.items():
    group = persona_metadata.get(pid, {}).get(group_attr)
    if group is None:
        continue
    grouped_results.setdefault(group, []).append(loss)

print(f"\n📁 Group-Level Fairness Loss (grouped by {group_attr}):")
for group, losses in grouped_results.items():
    print(f"{group:<15}: mean = {np.mean(losses):.4f}, max = {np.max(losses):.4f}")
