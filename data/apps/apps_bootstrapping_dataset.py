import json
from typing import Dict, List, Tuple, Union, Iterable

# from torch.utils.data import Dataset
from tqdm import tqdm

from data.apps.apps_dataset import APPSTask, construct_prompt_from_task, load_apps_dataset


class APPSCodePrompts():
    """
    Dataset for inference generating code answers.
    """
    def __init__(self, dataset: Iterable[APPSTask], few_shot_examples: str = "", num_of_tests: int = 1,
                 example_test_cases: dict = None, require_example_test_cases: bool = False,
                 load_valid_dataset: bool = False, testing: bool = False,
                 valid_ids_filepath: str= "data/apps/valid_problem_ids.json",
                 remove_test_examples_from_prompt: bool = False):
        """
        Initializes the dataset.
        :param dataset: The APPS dataset split.
        :param few_shot_examples: The few shot examples to use (plan to use none, because of the model's context window).
        :param num_of_tests: The number of tests to use to generate the prompt.
        :param example_test_cases: The example test cases extracted from the question using "apps_extract_example_test.ipynb".
        If None, then no test cases will be added to the tasks resulting from this dataset's __getitem__ function.
        :param require_example_test_cases: Whether to throw away tasks that do not have example test cases.
        """
        self.dataset = dataset
        self.few_shot_examples = few_shot_examples
        self.num_of_tests = num_of_tests
        self.example_test_cases = example_test_cases
        self.require_example_test_cases = require_example_test_cases
        self.remove_test_examples_from_prompt = remove_test_examples_from_prompt

        self.load_valid_dataset = load_valid_dataset
        with open(valid_ids_filepath, "r", encoding="utf-8") as f:
            self.valid_problem_ids = set(json.load(f))

        self.testing = testing

        # Prepare the dataset in advance
        self.prepared_dataset = self.prepare_data()

    def prepare_data(self):
        prepared_dataset = []

        for _, task in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            # Skip tasks that are not in the valid problem ids, if load_valid_dataset is True.
            # else skip tasks that are in the valid problem ids to construct the train dataset.
            if not self.testing and (
                    (self.load_valid_dataset and task["problem_id"] not in self.valid_problem_ids) or
                    (not self.load_valid_dataset and task["problem_id"] in self.valid_problem_ids)
            ):
                continue

            # Apparently saving JSON files with int keys converts them to strings? Therefore we will index with strings.
            if self.require_example_test_cases and self.example_test_cases and self.example_test_cases[str(task["problem_id"])]:
                # Make a subselection of the example test cases based on the self.num_of_tests parameter
                in_out_dict = {k: v[:self.num_of_tests] for k, v in self.example_test_cases[str(task["problem_id"])].items()}

                # Add the example test cases to the task
                task["example_test_case"] = in_out_dict

                # Generate prompt
                prompt = self.few_shot_examples + construct_prompt_from_task(task)

                # Add to the dataset we're preparing
                prepared_dataset.append((prompt, task))
            elif not self.require_example_test_cases:
                # Generate prompt
                prompt = self.few_shot_examples + construct_prompt_from_task(task,
                                                                             remove_example_test=self.remove_test_examples_from_prompt)

                # Add to the dataset we're preparing
                prepared_dataset.append((prompt, task))

        return prepared_dataset

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.prepared_dataset)

    def __getitem__(self, index) -> Tuple[str, APPSTask]:
        """
        Returns the item at the given index.
        :param index: The index of the item.
        :return: The string at the given index.
        """
        return self.prepared_dataset[index]



if __name__ == "__main__":
    # For testing.

    dataset = load_apps_dataset()

    train_data = dataset["train"]
    print(f"Train data size: {len(train_data)}")

    # valid_data = dataset["validation"]
    # print(f"Valid data size: {len(valid_data)}")

    test_data = dataset["test"]
    print(f"Test data size: {len(test_data)}")

    # Import test examples and see if they're processed correctly
    with open("./example_test_cases/train.json", "r", encoding="utf-8") as f:
        example_test_cases = json.load(f)

    number_of_solutions = []
    for train_example in train_data:
        number_of_solutions.append(len(json.loads(train_example["solutions"])))

    # Scipy describe the number of solutions
    # from scipy import stats
    # print(stats.describe(number_of_solutions))

    # Plot a boxplot using seaborn
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.boxplot(number_of_solutions)
    # plt.title("Number of solutions per task in training dataset")
    # plt.xlabel("Training dataset")
    # plt.ylabel("Number of solutions")
    # plt.ylim(0, 30)
    # plt.show()

    dataset = APPSCodePrompts(train_data, "", example_test_cases=example_test_cases, require_example_test_cases=True,
                              valid_ids_filepath="./valid_problem_ids.json")
    print(f"Dataset size: {len(dataset)}")
    print(dataset[1][1]["example_test_case"])
    print(dataset[1][0])
    print()
    print()
    print(json.loads(dataset[1][1]["solutions"])[0] + "\n[DONE]")
    print(f"problem_id: ", dataset[1][1]["problem_id"])
    print()

    print("Valid dataset")
    dataset = APPSCodePrompts(train_data, "", example_test_cases=example_test_cases, require_example_test_cases=True,
                              load_valid_dataset=True, valid_ids_filepath="./valid_problem_ids.json")
    print(f"Dataset size: {len(dataset)}")
    print(dataset[1][0])
    print()
    print()
    print(json.loads(dataset[1][1]["solutions"])[0] + "\n[DONE]")
    print(f"problem_id: ", dataset[1][1]["problem_id"])
    print(dataset[1][1]["example_test_case"])
    print(json.loads(dataset[1][1]["input_output"]))

    # See how many example test cases and hidden test cases overlap during training...

    # counter = 0
    # for prompt, task in tqdm(dataset):
    #     try:
    #         input_output = json.loads(task["input_output"])
    #     except:
    #         continue
    #
    #     example_and_hidden_are_equivalent = task["example_test_case"] == input_output
    #
    #     counter += example_and_hidden_are_equivalent
    #
    #     if not example_and_hidden_are_equivalent:
    #         print(input_output)
    #         print(task["example_test_case"])
    #         print()
    #         pass
    #
    # print(counter)

    # Import test examples and see if they're processed correctly
    with open("./example_test_cases/test.json", "r", encoding="utf-8") as f:
        example_test_cases = json.load(f)

    dataset = APPSCodePrompts(test_data, "", example_test_cases=example_test_cases, require_example_test_cases=True,
                              valid_ids_filepath="./valid_problem_ids.json")
    counter = 0
    for prompt, task in tqdm(dataset):
        try:
            input_output = json.loads(task["input_output"])
        except:
            continue

        example_and_hidden_are_equivalent = task["example_test_case"] == input_output

        counter += example_and_hidden_are_equivalent


        if not example_and_hidden_are_equivalent:
            if len(input_output["inputs"]) > 10:
                continue
            print(input_output)
            print(task["example_test_case"])
            print()

    print(counter)