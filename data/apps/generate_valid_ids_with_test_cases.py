import json
import random

# Set fixed seed for reproducibility.
random.seed(42)

from data.apps.apps_dataset import load_apps_dataset

dataset = load_apps_dataset()
train_data = dataset["train"]

with open("./example_test_cases/train.json", "r", encoding="utf-8") as f:
    example_test_cases = json.load(f)

ids_with_example_test_cases = []

for task in train_data:
    if example_test_cases[str(task["problem_id"])]:
        ids_with_example_test_cases.append(task["problem_id"])

print(f"Number of tasks with example test cases: {len(ids_with_example_test_cases)}")

# Choosing 600 problems at random (as described in the CodeRanker paper).
random.shuffle(ids_with_example_test_cases)
ids_with_example_test_cases = ids_with_example_test_cases[:600]

# Save to json file.
with open("./valid_problem_ids.json", "w", encoding="utf-8") as f:
    json.dump(ids_with_example_test_cases, f)
