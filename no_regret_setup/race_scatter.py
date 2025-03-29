import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from no_regret import FairnessElicitationAlgorithm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load shared data outside the parallel loop
with open("multi_persona_data/persona_metadata.json", "r") as f:
    persona_metadata = json.load(f)

with open("multi_persona_data/final_train.json", "r") as f:
    train_constraints_raw = json.load(f)

with open("multi_persona_data/final_holdout.json", "r") as f:
    test_constraints_raw = json.load(f)


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
id_to_idx = {id_val: idx for idx, id_val in enumerate(ids_test)}

# Define function to run per persona
def train_and_evaluate(train_persona_id):
    train_constraints = train_constraints_raw[train_persona_id]
    test_constraints = [
        constraint for pid, constraints in test_constraints_raw.items()
        if pid != train_persona_id for constraint in constraints
    ]

    algo = FairnessElicitationAlgorithm(
        data_path="train200_subset.parquet",
        constraint_sets_path="multi_persona_data/final_train.json",
        time_horizon=300,
        C_lambda=10.0,
        C_tau=1.0
    )
    algo.load_constraint_sets(custom_constraints=train_constraints)
    result = algo.run(gamma_values=[0.15])

    fairness_losses = []
    max_violation = 0
    for persona_id, persona_constraints in test_constraints_raw.items():
        if persona_id == train_persona_id:
            continue
        for constraint in persona_constraints:
            i_id, j_id = constraint["pair"]
            if i_id in id_to_idx and j_id in id_to_idx:
                i = id_to_idx[i_id]
                j = id_to_idx[j_id]
                x_i = X_test[i].reshape(1, -1)
                x_j = X_test[j].reshape(1, -1)
                prob_i = result[0.15]["final_model"].predict_proba(x_i)[0, 1]
                prob_j = result[0.15]["final_model"].predict_proba(x_j)[0, 1]
                violation = max(0, prob_i - prob_j - 0.05)
                fairness_losses.append(violation)
                max_violation = max(max_violation, violation)
    
    avg_fairness_loss = np.mean(fairness_losses)
    y_pred = result[0.15]["final_model"].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    race = persona_metadata[train_persona_id]["race"]

    return (max_violation, avg_fairness_loss, accuracy, race)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
# Run in parallel
results = []
# Track start time
start_time = time.time()

persona_ids = list(train_constraints_raw.keys())
with ProcessPoolExecutor() as executor:

    futures = {executor.submit(train_and_evaluate, pid): pid for pid in persona_ids}
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Personas processed")):
        result = future.result()
        results.append(result)

        # ETA tracking
        elapsed = time.time() - start_time
        avg_time_per_persona = elapsed / (i + 1)
        remaining = avg_time_per_persona * (len(futures) - (i + 1))
        print(f"✔ Processed {i+1}/{len(futures)} | ⏱ Avg: {avg_time_per_persona:.2f}s | ⌛ ETA: {remaining/60:.1f} min")

# Unpack results
max_fairness_losses, avg_fairness_losses, accuracies, training_races = zip(*results)

# Plotting
TOL_COLORS = ["#117733", "#332288", "#44AA99", "#88CCEE", "#DDCC77"]
race_colors = {'White': TOL_COLORS[0], 'Black': TOL_COLORS[1], 'Asian': TOL_COLORS[2], 'Other / Mixed': TOL_COLORS[3], 'American Indian or Alaska Native': TOL_COLORS[4]}
colors = [race_colors[r] for r in training_races]

plt.figure(figsize=(10, 6))
plt.scatter(max_fairness_losses, accuracies, c=colors, alpha=0.7)
plt.xlabel("Max Fairness Loss")
plt.ylabel("Accuracy")
plt.title("Max Fairness Loss vs Accuracy")
plt.grid(True)
plt.savefig("0.15_max_fairness_vs_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(avg_fairness_losses, accuracies, c=colors, alpha=0.7)
plt.xlabel("Average Fairness Loss")
plt.ylabel("Accuracy")
plt.title("Average Fairness Loss vs Accuracy")
plt.grid(True)
plt.savefig("0.15_avg_fairness_vs_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()
