import json
import random
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from no_regret import FairnessElicitationAlgorithm  # update to your actual import
from testing import compute_persona_fairness_losses, summarize_by_group  # same here

# ------------------------
# Load Data and Metadata
# ------------------------

with open("multi_persona_data/persona_metadata.json") as f:
    persona_metadata = json.load(f)
with open("multi_persona_data/final_train.json") as f:
    full_train_constraints = json.load(f)
with open("multi_persona_data/final_holdout.json") as f:
    final_holdout_constraints = json.load(f)

# ------------------------
# Sample 10 personas per race group
# ------------------------

random.seed(42)
race_to_personas = defaultdict(list)
for pid, meta in persona_metadata.items():
    race = meta.get("race")
    if race in {"White", "Black or African American", "Asian"}:
        race_to_personas[race].append(pid)

group_sample = {
    race: random.sample(personas, 10)
    for race, personas in race_to_personas.items()
    if len(personas) >= 10
}

# ------------------------
# Load and process test data
# ------------------------

test_df = pd.read_parquet("test1000_subset.parquet")
y_test = test_df['two_year_recid'].values
ids_test = test_df['id'].tolist()

with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("encoded_feature_names.pkl", "rb") as f:
    encoded_feature_names = pickle.load(f)
with open("train_feature_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

categorical_columns = ['sex', 'race', 'c_charge_degree']
categorical_data = test_df[categorical_columns].fillna("Unknown")
encoded_test = encoder.transform(categorical_data)

test_df_proc = test_df.drop(columns=categorical_columns + ['two_year_recid'])
encoded_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=test_df_proc.index)
test_df_proc = pd.concat([test_df_proc, encoded_df], axis=1)

for col in train_columns:
    if col not in test_df_proc.columns:
        test_df_proc[col] = 0
test_df_proc = test_df_proc[train_columns]
test_df_proc = test_df_proc.apply(pd.to_numeric, errors='coerce').fillna(0)

X_test = test_df_proc.values

# ------------------------
# Evaluate models trained on each group
# ------------------------

gamma = 0.0
results = {}

for train_race, train_personas in group_sample.items():
    print(f"\n🔧 Training no-regret model on constraints from {train_race}...")

    # --- 1. Collect constraints from selected personas
    selected_constraints = []
    for pid in train_personas:
        selected_constraints += full_train_constraints.get(pid, [])

    # --- 2. Train model using your FairnessElicitationAlgorithm
    fea = FairnessElicitationAlgorithm(
        data_path="train200_subset.parquet",
        constraint_sets_path="multi_persona_data/final_train.json"
    )
    fea.load_constraint_sets(custom_constraints=selected_constraints)

    # Run with your chosen gamma
    result_dict = fea.run(gamma_values=[gamma])
    final_model = result_dict[gamma]['final_model']

    # --- 3. Predict on test set
    probs_test = final_model.predict_proba(X_test)[:, 1]
    probs_by_id = {id_: prob for id_, prob in zip(ids_test, probs_test)}

    # --- 4. Compute fairness loss on *all personas*
    persona_losses = compute_persona_fairness_losses(probs_by_id, final_holdout_constraints, gamma)
    race_stats = summarize_by_group(persona_losses, persona_metadata, "race")

    results[train_race] = race_stats

# ------------------------
# Output Table
# ------------------------

print("\n📊 Cross-Group Fairness Loss Summary (γ = 0.1):")
races = list(group_sample.keys())
header = "Trained On → Evaluated On".ljust(30) + "  " + "  ".join(f"{r:25s}" for r in races)
print(header)
for train_race in races:
    row = f"{train_race:30s}  "
    for eval_race in races:
        mean_loss = results[train_race].get(eval_race, {}).get("mean", float('nan'))
        row += f"{mean_loss:25.4f}  "
    print(row)
