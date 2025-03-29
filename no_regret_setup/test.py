from run_experiments import train_and_evaluate  # or wherever you defined it
import numpy as np
import json

# Load metadata
with open("multi_persona_data/persona_metadata.json", "r") as f:
    metadata = json.load(f)

# Pick one judge with known metadata
judge_id = list(metadata.keys())[5]  # or "0001", etc.
print(judge_id)
# Define gamma sweep
gamma_sweep = [0.0, 0.1, 0.2]

# Run the test
results = train_and_evaluate(
    train_judge_ids=[judge_id],
    train_constraints_per_judge=50,
    gamma_train=0.1,
    gamma_test_values=gamma_sweep,
    label=f"test_run_{judge_id}",
    metadata=metadata
)

# Check results
import pandas as pd
df = pd.DataFrame(results)
print(df)
