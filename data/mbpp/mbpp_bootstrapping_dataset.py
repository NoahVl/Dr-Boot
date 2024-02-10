from typing import Dict, List, Tuple, Union

import torch
import transformers

from torch.utils.data import Dataset
from tqdm import tqdm

from data.mbpp.mbpp_dataset import load_mbpp_dataset, construct_prompt_from_task, MBPPTask


class MBPPCodePrompts(Dataset):
    """
    Dataset for inference generating code answers.
    """

    def __init__(self, dataset: List[MBPPTask], few_shot_examples: str, num_of_tests: int = 1):
        """
        Initializes the dataset.
        :param dataset: The MBPP dataset split.
        """
        self.dataset = dataset
        self.few_shot_examples = few_shot_examples
        self.num_of_tests = num_of_tests

        # Prepare the dataset in advance
        self.prepared_dataset = self.prepare_data()

    def prepare_data(self):
        prepared_dataset = []

        for _, task in tqdm(enumerate(self.dataset)):
            prompt = self.few_shot_examples + construct_prompt_from_task(task, num_of_tests=self.num_of_tests)

            prepared_dataset.append((prompt, task))

        return prepared_dataset

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.prepared_dataset)

    def __getitem__(self, index) -> Tuple[str, MBPPTask]:
        """
        Returns the item at the given index.
        :param index: The index of the item.
        :return: The string at the given index.
        """
        return self.prepared_dataset[index]


class PlainBootstrappingDataset(Dataset):
    """
    Dataset for finetuning CodeT5-NTP on the MBPP dataset.
    """

    def __init__(self, correct_code_with_prompt_and_task: List[Tuple[str, str, MBPPTask]], tokenizer: transformers.PreTrainedTokenizer,
                 max_source_length: int = 512, max_target_length: int = 512, left_truncate=False):
        """
        Initializes the dataset.
        :param dataset: The MBPP dataset.
        :param tokenizer: The tokenizer to use.
        :param max_source_length: The maximum length of the source.
        :param max_target_length: The maximum length of the target.
        """
        self.dataset = correct_code_with_prompt_and_task
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.left_truncate = left_truncate

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """
        Returns the item at the given index.
        :param index: The index of the item.
        :return: a dict of the form: { "input_ids" : torch.LongTensor, "labels" :  torch.LongTensor}
        supposedly compatible with the Huggingface Trainer.
        """
        input_ids = []
        target_ids = []

        code, prompt, task = self.dataset[index]

        # We only pack one sample, because we are using a T5 model.
        cur_prompt_ids = self.tokenizer.encode(prompt)
        cur_solution_ids = self.tokenizer.encode(code.strip() + "\n[DONE]")

        input_ids.extend(cur_prompt_ids)

        start_index = len(input_ids) - self.max_source_length

        # Used to be: this was very likely a bug
        # start_index = self.max_source_length - len(cur_prompt_ids)  # To keep the last self.max_source_length tokens

        if start_index > 0 and ("Feedback:" in prompt or self.left_truncate):  # Only left truncate while repairing.
            input_ids = input_ids[start_index:]

        target_ids.extend(cur_solution_ids)

        # Add end of sentence token to fill up the rest of the input.
        input_ids.extend([self.tokenizer.eos_token_id] * self.max_source_length)

        # Add the token that prevents loss during training from being calculated for the padding tokens.
        target_ids.extend([-100] * self.max_target_length)

        # Cut off the excess
        input_ids = input_ids[:self.max_source_length]
        target_ids = target_ids[:self.max_target_length]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(target_ids)
        }


if __name__ == "__main__":
    # For testing

    dataset = load_mbpp_dataset()

    train_data = dataset["train"]
    print(f"Train data size: {len(train_data)}")

    valid_data = dataset["validation"]
    print(f"Valid data size: {len(valid_data)}")

    test_data = dataset["test"]
    print(f"Test data size: {len(test_data)}")

    dataset = MBPPCodePrompts(train_data, "")
    print(f"Dataset size: {len(dataset)}")
    print(dataset[1][0])
    print()
    print(dataset[1][1]["code"] + "\n[DONE]")
