import json
from typing import Dict, Literal, Iterable

import torch
import transformers

from torch.utils.data import Dataset
from tqdm import tqdm

from data.apps.apps_dataset import load_apps_dataset, construct_prompt_from_task
from data.utils import reindent_code


class APPSFinetuningDataset(Dataset):
    """
    Dataset for finetuning CodeT5-NTP on the APPS dataset.
    """
    def __init__(self, dataset: Iterable, tokenizer: transformers.PreTrainedTokenizer,
                 max_source_length: int = 512, max_target_length: int = 512, load_valid_dataset=False,
                 example_test_cases=None, require_example_test_cases=False, remove_example_test_cases=False):
        """
        Initializes the dataset.
        :param dataset: The APPS dataset.
        :param tokenizer: The tokenizer to use.
        :param max_source_length: The maximum length of the source.
        :param max_target_length: The maximum length of the target.
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.load_valid_dataset = load_valid_dataset

        self.example_test_cases = example_test_cases
        self.require_example_test_cases = require_example_test_cases
        self.remove_example_test_cases = remove_example_test_cases

        with open("data/apps/valid_problem_ids.json", "r", encoding="utf-8") as f:
            self.valid_problem_ids = set(json.load(f))

        self.prepared_dataset = self.prepare_data(dataset)

    def prepare_data(self, dataset):
        prepared_dataset = []

        for _, task in tqdm(enumerate(dataset), total=len(dataset)):
            # Skip tasks that are not in the valid problem ids, if load_valid_dataset is True.
            # else skip tasks that are in the valid problem ids to construct the train dataset.
            if (self.load_valid_dataset and task["problem_id"] not in self.valid_problem_ids) or \
                (not self.load_valid_dataset and task["problem_id"] in self.valid_problem_ids) or \
                (self.require_example_test_cases and self.example_test_cases and
                 not self.example_test_cases[str(task["problem_id"])]):
                continue

            prepared_dataset.append(task)

        return prepared_dataset

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.prepared_dataset)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """
        Returns the item at the given index.
        :param index: The index of the item.
        :return: a dict of the form: { "input_ids" : torch.LongTensor, "labels" :  torch.LongTensor}
        supposedly compatible with the Huggingface Trainer.
        """
        input_ids = []
        target_ids = []

        # We only pack one sample, because we are using a T5 model.
        cur_prompt_ids = self.tokenizer.encode(construct_prompt_from_task(self.prepared_dataset[index],
                                                                          remove_example_test=self.remove_example_test_cases))
        cur_solution_ids = self.tokenizer.encode(reindent_code(json.loads(self.prepared_dataset[index]["solutions"])[0]).strip() + "\n[DONE]")

        input_ids.extend(cur_prompt_ids)

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
    pass