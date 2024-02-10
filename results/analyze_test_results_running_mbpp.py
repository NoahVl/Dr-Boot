# Import the jsons from ./experiment_results/apps_test/ and store them in a dict with their filename as the key.

import json
import os
from collections import defaultdict

import numpy as np

experiments = {}

for filename in os.listdir("./experiment_results/mbpp_runs/"):
    with open(os.path.join("./experiment_results/mbpp_runs/", filename), "r", encoding="utf-8") as f:
        experiments[filename.split(".json")[0]] = json.load(f)

# Map model names to best performing model entry that we judged from the validation set, that we later evaluated, so we only analyze those results
model_name_to_best_model = {
    "plain_bootstrap_mbpp_1": 5,
    "plain_bootstrap_mbpp_2": 1,
    "plain_bootstrap_mbpp_3": 2,
    "simple_feedback_mbpp_1": 4,
    "simple_feedback_mbpp_2": 1,
    "simple_feedback_mbpp_3": 9,
    "full_feedback_mbpp_1": 5,
    "full_feedback_mbpp_2": 8,
    "full_feedback_mbpp_3": 3,
}

selected_experiments = {

}
# First select only the best_model entry for every experiment in the experiments dict
for model_name, loaded_json in experiments.items():
    selected_experiments[model_name] = {key: value[model_name_to_best_model[model_name]] for
                                        key, value in loaded_json.items() if "test" in key}


# Collect the results per shared training objective
training_objectives = ["plain_bootstrap", "simple_feedback", "full_feedback"]
collected_experiments = {}
for training_objective in training_objectives:
    collected_experiments[training_objective] = defaultdict(list)

    for model_name, loaded_json in selected_experiments.items():
        if training_objective in model_name:
            for key, value in loaded_json.items():
                collected_experiments[training_objective][key].append(value)


# Calculate mean and std for the experiments
for training_objective, experiments in collected_experiments.items():
    print(f"{training_objective}:")

    for experiment_name, values in experiments.items():
        print(f"{experiment_name}, {np.mean(values):.2f} ± {np.std(values):.2f}")

    print("\n\n")