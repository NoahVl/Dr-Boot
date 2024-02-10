# Use seaborn to plot the results of the experiments
import os

import seaborn as sns
import json
import pandas as pd
from matplotlib import pyplot as plt


def remove_test_from_dict(dictionary: dict):
    # Delete all keys containing test
    keys = set(dictionary.keys())
    for key in keys:
        if "test" in key:
            dictionary.pop(key)


datasets = dict()

# Automate this by scanning the json files in the folder and loading them into a dictionary with their filename as
# key, minus the .json extension 1. Get all files in the folder
for file in os.listdir("./experiment_results/mbpp_runs"):
    # 2. Check if the file is a json file
    if file.endswith(".json"):
        # 3. Load the json file into a dictionary
        with open(os.path.join("./experiment_results/mbpp_runs", file), "r", encoding="utf-8") as f:
            datasets[file[:-5]] = json.load(f)

# for dataset in datasets.values():
#     remove_test_from_dict(dataset)

# Combine the data into a single dataframe
dataframe = pd.DataFrame({"dataset_name": pd.Series(dtype="str"),
                          "experiment": pd.Series(dtype="str"),
                          "step": pd.Series(dtype="int"),
                          "score": pd.Series(dtype="float")})

baseline_experiment_names = ["baseline_1_valid", "baseline_2_valid", "baseline_3_valid"]
greedy_baseline_and_simple_feedback = ["baseline_1_valid", "simple_feedback_valid", "simple_feedback_repair_valid"]
greedy_baseline_and_full_feedback = ["baseline_1_valid", "full_feedback_valid", "full_feedback_repair_valid"]

baseline_2_and_simple_feedback = ["baseline_2_valid", "simple_feedback_valid", "simple_feedback_repair_valid"]
baseline_2_and_full_feedback = ["baseline_2_valid", "full_feedback_valid", "full_feedback_repair_valid"]

simple_feedback_experiment = ["simple_feedback_valid", "simple_feedback_repair_valid"]
full_feedback_experiment = ["full_feedback_valid", "full_feedback_repair_valid"]


### CHANGE THIS ###
# target_experiments = baseline_experiment_names
# #
# # Plain baselines
# for dataset_name in [key for key in datasets.keys() if "plain" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in target_experiments:
#         for step, score in enumerate(dataset[experiment]):
#             experiment_name = "Baseline 1: Pass@1" if "baseline_1" in experiment else "Baseline 2: Pass@2" if "baseline_2" in experiment else "Baseline 3: Pass@2"
#             new_row = [dataset_name, experiment_name, step, score]
#             dataframe.loc[len(dataframe)] = new_row
# plt.title("Plain bootstrapping model results on validation dataset")
# sns.lineplot(data=dataframe, x="step", y="score", hue="experiment")
# plt.xlabel("Bootstrapping step")
# plt.ylabel("Pass@k score")
# plt.ylim(0, 40)
# plt.savefig('plot.pdf')
# plt.show()

# Comparison of simple model vs baseline
# for dataset_name in [key for key in datasets.keys() if "plain" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in ["baseline_2_valid"]:
#         for step, score in enumerate(dataset[experiment]):
#             new_row = [dataset_name, experiment, step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
# for dataset_name in [key for key in datasets.keys() if "simple" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in simple_feedback_experiment:   # Change me! This specifies the experiments to plot.
#         for step, score in enumerate(dataset[experiment]):
#             new_row = [dataset_name, experiment, step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
#
# # print(dataframe)
# sns.lineplot(data=dataframe, x="step", y="score", hue="experiment")
# plt.ylim(0, 30)
# plt.title("Plain baseline vs. simple feedback model results on validation dataset")
# plt.show()



# Comparison of simple or full model vs baseline
# for dataset_name in [key for key in datasets.keys() if "plain" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in ["baseline_1_valid"]:
#         for step, score in enumerate(dataset[experiment]):
#             new_row = [dataset_name, "Plain: Baseline 1, Pass@1", step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
# for dataset_name in [key for key in datasets.keys() if "simple" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in simple_feedback_experiment:  # Change me! This specifies the experiments to plot.
#         for step, score in enumerate(dataset[experiment]):
#             experiment_name = "Simple Feedback: Pass@1" if "repair" not in experiment else "Simple Feedback: Edit Pass@1"
#             new_row = [dataset_name, experiment_name, step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
#
# # print(dataframe)
# sns.lineplot(data=dataframe, x="step", y="score", hue="experiment")
# plt.ylim(0, 40)
# plt.ylabel("Pass@k score")
# plt.xlabel("Bootstrapping step")
# plt.title("Plain baseline vs. simple feedback model results on validation dataset")
# plt.savefig('plot.pdf')
# plt.show()

# Bar chart of max performance per experiment type.
# for dataset_name in [key for key in datasets.keys() if "plain" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in ["baseline_2_valid"]:
#         for step, score in enumerate(dataset[experiment]):
#             new_row = [dataset_name, experiment, step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
# for dataset_name in [key for key in datasets.keys() if "simple" in key and "full" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in simple_feedback_experiment:
#         for step, score in enumerate(dataset[experiment]):
#             new_row = [dataset_name, experiment+f"_{dataset_name.split('_')[0]}_model", step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
#
# # print(dataframe)
# sns.barplot(data=dataframe, x="experiment", y="score", hue="experiment")
# plt.ylim(0, 30)
# plt.title("Baseline vs. simple vs. full feedback with simple feedback repairing")
# plt.show()

# # Comparison of simple model vs full model
# for dataset_name in [key for key in datasets.keys() if "simple" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in ["simple_feedback_repair_valid"]:
#         for step, score in enumerate(dataset[experiment]):
#             experiment_name = "Simple Feedback: Pass@1" if "repair" not in experiment else "Simple Feedback: Edit Pass@1"
#             new_row = [dataset_name, experiment_name, step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
# for dataset_name in [key for key in datasets.keys() if "full" in key]:
#     dataset = datasets[dataset_name]
#     for experiment in ["full_feedback_repair_valid"]:  # Change me! This specifies the experiments to plot.
#         for step, score in enumerate(dataset[experiment]):
#             experiment_name = "Full Feedback: Pass@1" if "repair" not in experiment else "Full Feedback: Edit Pass@1"
#             new_row = [dataset_name, experiment_name, step, score]
#             dataframe.loc[len(dataframe)] = new_row
#
#
# # print(dataframe)
# sns.lineplot(data=dataframe, x="step", y="score", hue="experiment")
# plt.ylim(0, 40)
# plt.ylabel("Pass@k score")
# plt.xlabel("Bootstrapping step")
# plt.title("Simple feedback vs. full feedback model results on validation dataset")
# plt.savefig('plot.pdf')
# plt.show()





# Comparison of simple or full model vs baseline
for dataset_name in [key for key in datasets.keys() if "plain" in key]:
    dataset = datasets[dataset_name]
    for experiment in ["baseline_1_valid"]:
        for step, score in enumerate(dataset[experiment]):
            new_row = [dataset_name, "Plain: Baseline 1, Pass@1", step, score]
            dataframe.loc[len(dataframe)] = new_row

# Comparison of simple model vs full model
for dataset_name in [key for key in datasets.keys() if "simple" in key]:
    dataset = datasets[dataset_name]
    for experiment in ["simple_feedback_valid"]:
        for step, score in enumerate(dataset[experiment]):
            experiment_name = "Simple Feedback: Pass@1" if "repair" not in experiment else "Simple Feedback: Edit Pass@1"
            new_row = [dataset_name, experiment_name, step, score]
            dataframe.loc[len(dataframe)] = new_row

for dataset_name in [key for key in datasets.keys() if "full" in key]:
    dataset = datasets[dataset_name]
    for experiment in ["full_feedback_valid"]:  # Change me! This specifies the experiments to plot.
        for step, score in enumerate(dataset[experiment]):
            experiment_name = "Full Feedback: Pass@1" if "repair" not in experiment else "Full Feedback: Edit Pass@1"
            new_row = [dataset_name, experiment_name, step, score]
            dataframe.loc[len(dataframe)] = new_row


# print(dataframe)
sns.lineplot(data=dataframe, x="step", y="score", hue="experiment")
plt.ylim(0, 40)
plt.ylabel("Pass@k score")
plt.xlabel("Bootstrapping step")
plt.title("All models non-repairing pass@1 results on validation dataset")
plt.savefig('plot.pdf')
plt.show()