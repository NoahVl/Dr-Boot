from dataclasses import dataclass
from typing import Sequence, List, TypedDict

from datasets import load_dataset


# Dataclass for the mbpp tasks, they contain the following keys:
# dict_keys(['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list'])

class MBPPTask(TypedDict):
    task_id: int  # The ID of the task
    text: str  # The textual programming assignment
    code: str  # The code solution
    test_list: List[str]  # The list of asserts
    test_setup_code: str  # Predefined code for the asserts
    challenge_test_list: List[str]  # The list of asserts for the extra challenges


def load_mbpp_dataset():
    return load_dataset("mbpp")


def load_debug_mdpp_dataset(length: int = 10, split: str = "train"):
    return load_dataset("mbpp", split=f"{split}[:{length}]")


def construct_prompt_from_task(task: dict, num_of_tests=1) -> str:
    """
    Constructs the prompt for the model, loosely based on the format of the MBPP dataset's paper:
    "Program Synthesis with Large Language models", Austin et al. 2021," from an individual task.
    """
    code_assignment, tests = task["text"], task["test_list"]
    formatted_tests = "\n".join(tests[:num_of_tests])

    prompt = \
        f"""{code_assignment}
        
Your code should pass these tests:
{formatted_tests}

[ANSWER]
"""
    return prompt

def construct_initial_prompt(prompt_split, task_ids: Sequence = (2, 3, 4)) -> str:
    """
    Constructs the initial prompt for the model using the format of the MBPP dataset's paper:
    "Program Synthesis with Large Language models", Austin et al. 2021,

    "
    You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:

    {tests}
    [ANSWER]
    {code}
    [DONE]
    "
    where the [ANSWER] and [DONE] tokens were used to delimit the model solution.
    :param prompt_split: The prompt split of the dataset.
    :param task_ids: The task IDs to use for the prompts.
    :return: The initial prompt for the model.
    """
    # Initialize the prompt.
    prompt = ""

    for task_id, task in enumerate(prompt_split):
        if not task_ids or task_id in task_ids:
            code_assignment, tests, code = task["text"], task["test_list"], task["code"]
            formatted_tests = "\n".join(tests[:2])

            prompt += \
                f"""{code_assignment}

Your code should pass these tests:
{formatted_tests}

[ANSWER]
{code}
[DONE]

"""

    return prompt


def print_split_to_txt(split, file_path):
    """
    Prints the split to a txt file for manual inspection.
    :param split: The split of the mbpp dataset
    :param file_path: The filepath to save to.
    """

    with open(file_path, "w", encoding="utf-8") as f:
        for task in split:
            task: MBPPTask

            f.write(f"Task ID {task['task_id']}\n")
            f.write(f"{task['text']}\n\n")
            f.write(f"{task['test_list']}\n\n")
            f.write(f"{task['code']}\n")
            f.write("-" * 50 + "\n\n")


if __name__ == "__main__":
    # For testing

    dataset = load_mbpp_dataset()

    # Print the length of all splits: train, validation, prompt, test
    print("Printing lengths of splits:")
    print(f"Train: {len(dataset['train'])}")
    print(f"Validation: {len(dataset['validation'])}")
    print(f"Prompt: {len(dataset['prompt'])}")
    print(f"Test: {len(dataset['test'])}")
    print("-" * 50 + "\n")

    print(type(dataset))
    print(dataset)
    print(dataset["train"][0].keys())
    print(dataset["train"][0])
    print(dataset["train"]["text"][0])
    print(dataset["train"]["code"][0])

    print(construct_prompt_from_task(dataset["train"][2]))
    print(dataset["train"][2]["task_id"])
