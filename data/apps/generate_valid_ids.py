import json
import random

# Set fixed seed for reproducibility.
random.seed(42)

from data.apps.apps_dataset import load_apps_dataset

dataset = load_apps_dataset()
train_data = dataset["train"]

problem_ids = [task["problem_id"] for task in train_data]

print(f"Number of training tasks: {len(problem_ids)}")

# Choosing 598 problems at random (as described in the Self-Edit paper).
random.shuffle(problem_ids)
ids_with_example_test_cases = problem_ids[:598]

# Save to json file.
with open("./valid_problem_ids.json", "w", encoding="utf-8") as f:
    json.dump(ids_with_example_test_cases, f)