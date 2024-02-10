# Import the jsons from ./experiment_results/apps_test/ and store them in a dict with their filename as the key.

import json
import os
from collections import defaultdict

import numpy as np

experiments = {}

for filename in os.listdir("./experiment_results/mbpp_test/"):
    with open(os.path.join("./experiment_results/mbpp_test/", filename), "r", encoding="utf-8") as f:
        experiments[filename.split(".json")[0]] = json.load(f)

# First collect the greedy experiments
experiments_of_interest = ["baseline", "greedy"]
selected_experiments = {}

for model_name, loaded_json in experiments.items():
    selected_experiments[model_name] = {key: value[0] for
                                        key, value in loaded_json.items() for experiment in experiments_of_interest if experiment in key and not "errors" in key}

    # Special case for baseline_3, as we want the sampled pass@2.
    if "baseline_3_test" in selected_experiments[model_name]:
        selected_experiments[model_name]["baseline_3_test"] = loaded_json["baseline_3_test"][0]["sampled"]["2"]

training_objectives = ["regular", "plain_bootstrap", "simple_bootstrap", "full_bootstrap"]

# Collect the results per shared training objective
collected_experiments = {}
for training_objective in training_objectives:

    collected_experiments[training_objective] = defaultdict(list)

    for model_name, loaded_json in selected_experiments.items():
        if training_objective in model_name:
            for key, value in loaded_json.items():
                collected_experiments[training_objective][key].append(value)

print(collected_experiments)

for training_objective, experiments in collected_experiments.items():
    print(f"{training_objective}:")

    for experiment_name, values in experiments.items():
        print(f"{experiment_name}, {np.mean(values):.2f} ± {np.std(values):.2f}")

    print("\n\n")