# Import the jsons from ./experiment_results/apps_test/ and store them in a dict with their filename as the key.

import json
import os
import numpy as np

experiments = {}

for filename in os.listdir("./experiment_results/apps_test/"):
    with open(os.path.join("./experiment_results/apps_test/", filename), "r", encoding="utf-8") as f:
        experiments[filename.split(".json")[0]] = json.load(f)

# First print the greedy performance
def extract_greedy_performance(loaded_json):
    return loaded_json["full_feedback_test_greedy_pass@1"][0], loaded_json["full_feedback_repair_test_greedy_repair@1"][0]

processed_experiments = {}

print("Greedy performance:")
print("Model name, Pass@1, Repair@1")
# Initialize the dict
for model_name, loaded_json in experiments.items():
    processed_experiments[model_name[:-2]] = [[], []]

# Populate the values
for model_name, loaded_json in experiments.items():
    pass_at_1, repair_at_1 = extract_greedy_performance(loaded_json)
    processed_experiments[model_name[:-2]][0].append(pass_at_1)
    processed_experiments[model_name[:-2]][1].append(repair_at_1)

# Calculate mean and std for the experiments
for model_name, values in processed_experiments.items():
    print(f"{model_name}, {np.mean(values[0]):.2f} ± {np.std(values[0]):.2f}, {np.mean(values[1]):.2f} ± {np.std(values[1]):.2f}")

print("\n\n")

print("Sampling performance:")
print("Estimated performance")

def extract_sampled_performance(loaded_json, sampled_or_estimated="estimated"):
    return loaded_json["full_feedback_test"][0][sampled_or_estimated], loaded_json["full_feedback_repair_test"][0][sampled_or_estimated]

# Initialize the dict
sampled_experiment = {}

for model_name, loaded_json in experiments.items():
    sampled_experiment[model_name[:-2]] = [[[] for _ in range(4)], [[] for _ in range(4)]]

# Populate the values
for model_name, loaded_json in experiments.items():
    pass_at_k_values, edit_at_k_values = extract_sampled_performance(loaded_json, sampled_or_estimated="estimated")

    for i, k in enumerate([1, 2, 5, 10]):
        # Initial pass
        sampled_experiment[model_name[:-2]][0][i].append(pass_at_k_values[str(k)])

        # Repairing pass
        sampled_experiment[model_name[:-2]][1][i].append(edit_at_k_values[str(k)])

print("Model name, \t\t\t\t\tPass@1, \tRepair@1, \tPass@2, \tRepair@2, \tPass@5, \tRepair@5, \tPass@10, \tRepair@10")
# Calculate mean and std for the experiments
for model_name, values in sampled_experiment.items():
    print(f"{model_name}, {np.mean(values[0][0]):.2f} ± {np.std(values[0][0]):.2f}, {np.mean(values[1][0]):.2f} ± {np.std(values[1][0]):.2f}, {np.mean(values[0][1]):.2f} ± {np.std(values[0][1]):.2f}, {np.mean(values[1][1]):.2f} ± {np.std(values[1][1]):.2f}, {np.mean(values[0][2]):.2f} ± {np.std(values[0][2]):.2f}, {np.mean(values[1][2]):.2f} ± {np.std(values[1][2]):.2f}, {np.mean(values[0][3]):.2f} ± {np.std(values[0][3]):.2f}, {np.mean(values[1][3]):.2f} ± {np.std(values[1][3]):.2f}")

print("\n\n")

