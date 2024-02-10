from typing import Dict, Literal

import torch
import transformers

from torch.utils.data import Dataset

from data.mbpp.mbpp_dataset import load_mbpp_dataset, construct_prompt_from_task
from data.utils import reindent_code


class MBPPFinetuningDataset(Dataset):
    """
    Dataset for finetuning CodeT5-NTP on the MBPP dataset.
    """

    def __init__(self, dataset: Dataset, tokenizer: transformers.PreTrainedTokenizer, few_shot_examples: str = "",
                 max_source_length: int = 512, max_target_length: int = 512):
        """
        Initializes the dataset.
        :param dataset: The MBPP dataset.
        :param tokenizer: The tokenizer to use.
        :param max_source_length: The maximum length of the source.
        :param max_target_length: The maximum length of the target.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.few_shot_examples = few_shot_examples

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

        # We only pack one sample, because we are using a T5 model.
        cur_prompt_ids = self.tokenizer.encode(self.few_shot_examples + construct_prompt_from_task(self.dataset[index]))
        cur_solution_ids = self.tokenizer.encode(reindent_code(self.dataset[index]["code"]).strip() + "\n[DONE]")

        input_ids.extend(cur_prompt_ids)

        start_index = len(input_ids) - self.max_source_length

        # Used to be: this was very likely a bug
        # start_index = self.max_source_length - len(cur_prompt_ids)  # To keep the last self.max_source_length tokens

        if start_index > 0:
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

    from models.codet5_model import CodeT5Model

    dataset = load_mbpp_dataset()
    model = CodeT5Model(dataset["prompt"])

    finetuning_dataset = MBPPFinetuningDataset(dataset["train"], model.tokenizer)
    print(finetuning_dataset[0])