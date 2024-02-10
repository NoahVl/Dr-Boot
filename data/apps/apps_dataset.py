import json
from enum import Enum
from itertools import chain
from typing import TypedDict

# import scipy.stats
from datasets import load_dataset
from transformers import AutoTokenizer


class CODE_TYPE(Enum):
    """
    From CodeRL
    """
    call_based = 0
    standard_input = 1


class APPSTask(TypedDict):
    problem_id: int
    question: str
    solutions: str  # Needs to be loaded as json
    input_output: str  # Needs to be loaded as json, dict with inputs and outputs, sometimes fn_name.
    difficulty: str  # introductory, interview, competition
    url: str
    starter_code: str  # Should be included in the prompt, rarely present


def load_debug_apps_dataset(length: int = 10, split: str = "train"):
    return load_dataset("codeparrot/apps", split=f"{split}[:{length}]")


def load_apps_dataset():
    return load_dataset("codeparrot/apps")


def construct_prompt_from_task(task: APPSTask, remove_example_test: bool = False) -> str:
    """
    Constructs the prompt for the model using the format of the Measuring Coding Challenge Competence with APPS paper by
    Hendrycks et al. 2021," from an individual task.

    Instead of using Answer:, we use [ANSWER] and [DONE] to delimit the model solution.

    :param task: The task to construct the prompt from.
    :param remove_example_test: Whether to remove the example test from the prompt. This might be useful when
    the example tests are the same as the hidden tests. Defaults to False. Note: This will remove anything that comes
    after the examples, including the notes (as they write about the examples).
    """
    question_str = task["question"]
    question_str_lower = question_str.lower()

    if remove_example_test:
        # Remove everything after the introduction of examples.
        # TODO: To add if doing another run: example input, # examples,
        for keyword in ["-examples-", "-example-", "-example -", "example:", "example 1:", "examples:", "sample input"]:
            # Keywords taken from the CodeRL test unit cases extractor.
            if keyword in question_str_lower:
                keyword_index = question_str_lower.index(keyword)
                previous_newline_index = question_str_lower[:keyword_index].rfind("\n")
                question_str = question_str[:previous_newline_index].rstrip()

    prompt = \
        f"""Question:
{question_str}

"""
    try:
        in_outs = json.loads(task["input_output"])

        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
        else:
            which_type = CODE_TYPE.call_based  # Call-based
    except Exception as e:
        which_type = CODE_TYPE.standard_input  # Standard input

    if "starter_code" in task and task["starter_code"]:
        prompt += f"{task['starter_code']}\n"

    if which_type == CODE_TYPE.call_based:
        prompt += "\nUse Call-Based Format\n\n"
    else:
        prompt += "\nUse Standard Input Format\n\n"

    prompt += "[ANSWER]\n"

    return prompt



if __name__ == "__main__":
    # For testing.

    dataset = load_apps_dataset()
    print(dataset)
    print(dataset["train"][0].keys())

    first_example = dataset["train"][0]

    print(construct_prompt_from_task(first_example))
    print(json.loads(first_example["solutions"]))

    print("Done!")

    # Load CodeT5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py",
                                  truncation_side='left')

    tokenized_lengths_per_split = {dataset_name: [len(construct_prompt_from_task(task))for task in dataset[dataset_name]] for dataset_name in ["train", "test"]}
    tokenized_lengths = list(chain(*tokenized_lengths_per_split.keys()))

    # Plot boxplot of lengths using matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.boxplot(tokenized_lengths_per_split.values())
    ax.set_xticklabels(tokenized_lengths_per_split.keys())
    plt.title("Lengths of tokenized prompts for APPS dataset")
    plt.xlabel("Dataset split")
    plt.ylabel("Length (tokens)")
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(tokenized_lengths_per_split.values())
    ax.set_xticklabels(tokenized_lengths_per_split.keys())
    plt.title("Lengths of tokenized prompts for APPS dataset")
    plt.xlabel("Dataset split")
    plt.ylabel("Length (tokens)")
    plt.ylim(0, 2000)
    plt.show()

    # # In a new plot, restrict the y-axis to a reasonable range (0-1000)
    # plt.boxplot(tokenized_lengths)
    # plt.title("Lengths of tokenized prompts for APPS dataset")
    # plt.ylim(0, 1000)
    # plt.show()

    # Plot a bar-chart of each problem type
    problem_type_count = {"train_standard_input": 0, "train_call_based": 0,
                          "test_standard_input": 0, "test_call_based": 0}

    def get_problem_type(dataset_split):
        for task in dataset[dataset_split]:
            try:
                in_outs = json.loads(task["input_output"])

                if in_outs.get("fn_name") is None:
                    which_type = CODE_TYPE.standard_input  # Standard input
                else:
                    which_type = CODE_TYPE.call_based  # Call-based
            except Exception as e:
                which_type = CODE_TYPE.standard_input  # Standard input

            if which_type == CODE_TYPE.call_based:
                problem_type_count[f"{dataset_split}_call_based"] += 1
            else:
                problem_type_count[f"{dataset_split}_standard_input"] += 1

    get_problem_type("train")
    get_problem_type("test")

    print(problem_type_count)

    plt.bar(problem_type_count.keys(), problem_type_count.values())
    plt.title("Problem type counts for APPS dataset")
    plt.show()








