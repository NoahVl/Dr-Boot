import json
import os


def print_sankey(initial_errors_to_count: dict, repaired_errors_to_count: dict):
    for error, count in initial_errors_to_count.items():
        repaired_count = 0
        if error in repaired_errors_to_count:
            repaired_count = repaired_errors_to_count[error]
            print(f"{error} [{repaired_count}] Repaired")

        print(f"{error} [{count - repaired_count}] Unsuccessful Repair") if count - repaired_count > 0 else None

# Load some data
experiments = {}

for filename in os.listdir("./experiment_results/mbpp_test/"):
    with open(os.path.join("./experiment_results/mbpp_test/", filename), "r", encoding="utf-8") as f:
        experiments[filename.split(".json")[0]] = json.load(f)

# Collect the greedy experiments
target_file = "test_full_bootstrap_mbpp_3"
# print_sankey(experiments[target_file]["full_feedback_test_greedy_pass@1_errors_to_count"][0],
#              experiments[target_file]["full_feedback_repair_test_greedy_repair@1_errors_to_count"][0])

print_sankey(experiments[target_file]["full_feedback_test"][0]["errors_to_count"],
             experiments[target_file]["full_feedback_repair_test"][0]["repaired_errors_to_count"])

# Coloring
print(
"""
:Unsuccessful Repair #FF0000
:Repaired #32CD32
"""
)