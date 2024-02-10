# First party
import json
import os
from typing import Sequence, List

# Second party
from data.mbpp.mbpp_dataset import load_mbpp_dataset

# Third party
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Slightly faster on my laptop (RTX 3060-Mobile).
torch.set_float32_matmul_precision('high')


class CodeT5Model:
    """
    Class to load a CodeT5 model and predict code.
    """

    def __init__(self, compile_model=False, use_cuda_if_available=True,
                 model_path="Salesforce/codet5-large-ntp-py", local_files_only=False,
                 truncation_side="right"):
        """
        Initializes the model.
        :param prompt_split: The prompt split of the dataset. If not provided, no prompt will be added
        :param initial_prompt_ids: The task IDs to use for the initial prompt.
        :param use_cuda_if_available: Whether to use a CUDA device if available.
        :param model_path: The path to the model. Defaults to the large CodeT5-NTP model.
        But can be specified to import for example locally saved/fine-tuned models.
        """
        self.device = torch.device('cuda' if use_cuda_if_available and torch.cuda.is_available() else 'cpu')

        self.model = T5ForConditionalGeneration.from_pretrained(model_path,
                                                                local_files_only=local_files_only).to(self.device)

        # Set default truncation_side
        self.truncation_side = truncation_side

        # To load tokenizer: (as it is not finetuned and saved with the model)
        if local_files_only:
            # Get the name of the original model from the config by reading it as a json.
            config_path = os.path.join(model_path, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)

            print(f"Loading tokenizer from {config['_name_or_path']}")
            try:
                self.tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(config["_name_or_path"])
            except Exception:
                print("Failed to load tokenizer from original model, loading from default model: "
                      "Salesforce/codet5-large-ntp-py instead.")
                self.tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
        else:
            self.tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(model_path)


        # PyTorch 2.0 feature: Windows is currently not supported.
        if compile_model and os.name != 'nt':
            self.model: T5ForConditionalGeneration = torch.compile(self.model)
            print("Model compiled using PyTorch 2.0!")

    def predict_single_problem(self, assignment: str, max_sample_length=256, do_sample=False, batch_size=1) -> \
    List[str]:
        """
        Predicts the code for the given assignment.
        :param initial_prompt: The initial prompt to use.
        :param assignment: The assignment to predict_single_problem the code for.
        :param max_sample_length: The maximum length of the generated code.
        :param do_sample: Whether to sample from the model's output distribution, when False greedy decoding is used.
        :return: The predicted code.
        """
        input_ids = self.tokenizer(assignment, return_tensors="pt").input_ids.to(self.device) \
            .repeat(batch_size, 1)
        generated_ids = self.model.generate(input_ids, do_sample=do_sample, max_length=max_sample_length)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return output

    def predict_different_problems(self, assignments: list, max_sample_length=256, do_sample=False, max_length=600,
                                   num_beams=1, num_return_sequences=1, temperature=1.0, top_p=1.0) -> List[str]:
        # If we got the empty list, return the empty list.
        if not assignments:
            return []

        input_ids = self.tokenizer(assignments, padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt").input_ids.to(self.device)

        generated_ids = self.model.generate(input_ids, do_sample=do_sample, max_length=max_sample_length,
                                            num_beams=num_beams, num_return_sequences=num_return_sequences,
                                            temperature=temperature, top_p=top_p)

        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return output

    def put_model_on_cpu(self):
        self.model = self.model.to("cpu")
        return self

    def put_model_on_device(self):
        self.model = self.model.to(self.device)
        return self

    def change_tokenizer_truncation_direction(self, truncation_side="right"):
        self.tokenizer.truncation_side = truncation_side

    def reset_tokenizer_truncation_direction(self):
        self.tokenizer.truncation_side = self.truncation_side

if __name__ == "__main__":
    dataset = load_mbpp_dataset()
    model = CodeT5Model()

    test_prompt = \
        """Write a function that takes a list of numbers and returns the sum of the numbers. Your code should pass these tests:

assert sum_list([1, 2, 3]) == 6
assert sum_list([1, 2, 3, 4]) == 10
assert sum_list([20, 5, 5]) == 30
[BEGIN]
"""

    print(model.predict_single_problem(test_prompt))
