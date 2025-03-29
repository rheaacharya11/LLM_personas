import matplotlib.pyplot as plt
import json
from collections import defaultdict
import os
import numpy as np

# --- Load training data ---
with open("multi_persona_data/final_train.json", "r") as f:
    train_constraints_raw = json.load(f)

# --- Load persona metadata ---
with open("multi_persona_data/persona_metadata.json", "r") as f:
    persona_metadata = json.load(f)

# --- Create output folder ---
os.makedirs("histograms", exist_ok=True)

# --- Color palettes ---
TOL_COLORS = {
    "male": "#117733",
    "female": "#332288",
    "other": "#DDCC77",
}

RACE_COLORS = {
    "White": "#117733",
    "Black": "#332288",
    "Asian": "#44AA99",
    "Other / Mixed": "#88CCEE",
    "American Indian or Alaska Native": "#DDCC77",
    "Unknown": "#999999",
}

# --- Utility plotting function ---
def plot_normalized_histogram(data, title, xlabel, ylabel, color, filename, y_max=0.25, bins=15, xlim=None):
    if not data:
        return

    mean_val = np.mean(data)
    median_val = np.median(data)

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, density=True, alpha=0.9, color=color, edgecolor='black')

    plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    plt.axvline(median_val, color="blue", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")

    if xlim:
        plt.xlim(xlim)
    plt.ylim(0, y_max)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"histograms/{filename}.png")
    plt.show()

# --- GENDER-based constraint pair counts ---
gender_bins = {
    "male": [],
    "female": [],
    "other": [],
}

# --- RACE-based constraint pair counts ---
race_bins = defaultdict(list)

# --- Judges per constraint pair ---
judges_all = []
judges_by_gender = {
    "male": [],
    "female": [],
    "other": [],
}
judges_by_race = defaultdict(list)

# --- Populate all data buckets ---
for persona_id, constraints in train_constraints_raw.items():
    metadata = persona_metadata.get(persona_id, {})
    gender = metadata.get("sex", "unknown").lower()
    race = metadata.get("race", "Unknown")

    num_pairs = len(constraints)

    # Gender + race constraint pair count
    if gender == "male":
        gender_bins["male"].append(num_pairs)
    elif gender == "female":
        gender_bins["female"].append(num_pairs)
    else:
        gender_bins["other"].append(num_pairs)
    race_bins[race].append(num_pairs)

    # Judges per pair
    for pair in constraints:
        votes = pair.get("votes", [])
        n_judges = len(votes)
        judges_all.append(n_judges)

        # Gender
        if gender == "male":
            judges_by_gender["male"].append(n_judges)
        elif gender == "female":
            judges_by_gender["female"].append(n_judges)
        else:
            judges_by_gender["other"].append(n_judges)

        # Race
        judges_by_race[race].append(n_judges)

# --- Plot aggregated judges histogram ---
plot_normalized_histogram(
    data=judges_all,
    title="Judges per Constraint Pair (All)",
    xlabel="Number of Judges",
    ylabel="Proportion of Pairs",
    color="#444444",
    filename="judges_all",
    y_max=0.25,
    xlim=(0, max(judges_all) + 1)
)

# --- Plot by gender ---
for gender, data in judges_by_gender.items():
    plot_normalized_histogram(
        data=data,
        title=f"Judges per Constraint Pair ({gender.capitalize()})",
        xlabel="Number of Judges",
        ylabel="Proportion of Pairs",
        color=TOL_COLORS[gender],
        filename=f"judges_gender_{gender}",
        y_max=0.25,
        xlim=(0, max(judges_all) + 1)
    )

# --- Plot by race ---
for race, data in judges_by_race.items():
    safe_race_key = race.replace(" ", "_").replace("/", "_").lower()
    color = RACE_COLORS.get(race, "#999999")
    plot_normalized_histogram(
        data=data,
        title=f"Judges per Constraint Pair ({race})",
        xlabel="Number of Judges",
        ylabel="Proportion of Pairs",
        color=color,
        filename=f"judges_race_{safe_race_key}",
        y_max=0.25,
        xlim=(0, max(judges_all) + 1)
    )
