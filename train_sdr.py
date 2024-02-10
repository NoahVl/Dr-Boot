"""
This file contains the code for training a model using regular bootstrapping or
the bootstrapped SDR (self-debug-repair) method, with simple or full feedback for the repairing step.
This method is described in the thesis.
"""

import json
import os
import pprint
import shutil
from collections import defaultdict
from copy import deepcopy
import random
from typing import List, Tuple
from difflib import SequenceMatcher

import torch
import transformers
from datasets import Dataset
from tqdm import tqdm
from transformers import Trainer

from data.apps.apps_bootstrapping_dataset import APPSCodePrompts
from data.apps.apps_dataset import load_apps_dataset, load_debug_apps_dataset, APPSTask
from data.apps.apps_finetuning_dataset import APPSFinetuningDataset
from data.mbpp.mbpp_dataset import MBPPTask, load_mbpp_dataset, load_debug_mdpp_dataset
from data.mbpp.mbpp_bootstrapping_dataset import MBPPCodePrompts, PlainBootstrappingDataset
from data.mbpp.mbpp_finetuning_dataset import MBPPFinetuningDataset
from data.utils import reindent_code
from models.codet5_model import CodeT5Model
from test_mbpp_task import test_mbpp_task

# If on Windows, set the following to True.
if os.name != "nt":
    from test_apps_task import test_apps_task  # Windows does not support signals.
else:
    test_apps_task = None


###### HELPER FUNCTIONS ######

def find_error_name(prompt: str) -> str:
    """
    Finds the name of the error in the prompt.
    This code assumes that the prompt actually contains an error.

    :param prompt: The prompt to find the error name in.

    :return: The name of the error.
    """

    # Helper function for finding the name of the error in the feedback.
    # First find "Feedback: ", if Feedback is not present, return an empty string.
    # The new string will start after "Feedback: ".
    if "Feedback: " not in prompt:
        return ""

    prompt = prompt[prompt.rfind("Feedback: ") + len("Feedback: "):]

    # Split the string and see if there is a word that has Error in it.
    for word in prompt.split()[::-1]:
        if "error" in word.lower():
            # Usually the errors are formatted in such a way that "{error}:", we'd like to remove this trailing colon.
            if word[-1] == ':' in word:
                word = word[:-1]

            return word

    # Else, see if there is a string that says "we get the following output" for APPS
    # or: "The assertion is" for MBPP. If the pass_bool was indeed incorrect then these correspond to an OutputMismatchError.
    if "we get the following output" in prompt or "The assertion is" in prompt:
        return "OutputMismatchError"

    if "The code above timed out." in prompt:
        return "TimeoutError"

    print(f"UnsureError: {prompt}")

    return "UnsureError"


def calc_pass_at_k(generated_code_with_prompt_and_task: list, k: int, testing_function=test_mbpp_task):
    """
    Using the original pass_at_k definition to calculate the pass_at_k score.
    This is done during training to easily display the pass_at_k score, since we are doing greedy decoding.

    :param generated_code_with_prompt_and_task: The generated code with their respective prompt and task.
    :param k: The k to use for pass_at_k.
    :param testing_function: The testing function to use for testing the code.

    :return: The pass_at_k score.
    """
    ordered_per_k = [generated_code_with_prompt_and_task[i:i + k] for i in
                     range(0, len(generated_code_with_prompt_and_task), k)]

    correct_solutions = []

    for index, subresults in enumerate(ordered_per_k):
        for subindex, (code, prompt, task) in enumerate(subresults):
            pass_bool, feedback_string = testing_function(code, task)

            if pass_bool:
                correct_solutions.append((code, prompt, task))
                break

    return len(correct_solutions) / len(ordered_per_k) * 100


def log_examples(args, produced_code_with_prompt_and_task: List[Tuple[str, str, MBPPTask | APPSTask]],
                 correct_code_with_prompt_and_task: List[Tuple[str, str, MBPPTask | APPSTask]],
                 bootstrap_step: int, pass_at_one: float,
                 validation: bool = False,
                 test: bool = False):
    """
    Logs the examples to a file.

    :param args: The arguments.
    :param produced_code_with_prompt_and_task: All produced code with their respective prompt and task.
    :param correct_code_with_prompt_and_task: The correct codes with their respective prompt and task.
    :param bootstrap_step: The current bootstrap step.
    :param pass_at_one: The pass at 1 score.
    :param validation: Whether to write it as validation examples.
    :param test: Whether to write it as test examples.
    """

    if validation:
        file_name = f"valid_examples_{bootstrap_step}.txt"
    elif test:
        file_name = f"test_examples_{bootstrap_step}.txt"
    else:
        file_name = f"examples_{bootstrap_step}.txt"

    with open(os.path.join(args.save_dir, file_name), "w", encoding="utf-8") as f:
        # Write pass at 1 score
        f.write(f"Pass at 1: {pass_at_one}\n\n")

        for index, (code, prompt, task) in enumerate(produced_code_with_prompt_and_task):
            task_id = task["problem_id"] if "problem_id" in task else task["task_id"]
            f.write(f"Task {index}, id={task_id}:\n")
            f.write(
                f"{prompt[prompt.rfind('### Task Start ###') if 'Task Start' in prompt else 0:]}{code.strip()}\n[DONE]\n")
            f.write("-" * 100 + "\n\n\n\n")

        for index, (code, prompt, task) in enumerate(correct_code_with_prompt_and_task):
            task_id = task["problem_id"] if "problem_id" in task else task["task_id"]
            f.write("\n" * 10)
            f.write("Correct codes\n")
            f.write(f"Task {index}, id={task_id}:\n")
            f.write(
                f"Task: {prompt[prompt.rfind('### Task Start ###') if 'Task Start' in prompt else 0:]}{code.strip()}\n[DONE]\n")
            f.write("-" * 100 + "\n\n\n\n")

    print(f"Logged examples to {args.save_dir}/{file_name}")


def log_repaired_examples(args, second_incorrect_code_with_prompt_and_task, second_correct_code_with_prompt_and_task,
                          bootstrap_step, pass_with_repair, regular_pass_at_one, validation: bool = False,
                          test=False):
    """
    Logs the repaired examples to a file.

    :param args: The arguments.
    :param second_incorrect_code_with_prompt_and_task: The incorrect codes with their respective prompt and task.
    :param second_correct_code_with_prompt_and_task: The correct codes with their respective prompt and task.
    :param bootstrap_step: The current bootstrap step.
    :param pass_with_repair: The pass at 1 score with repair.
    :param regular_pass_at_one: The regular pass at 1 score.
    :param validation: Whether to write it as validation examples.
    :param test: Whether to write it as test examples.
    """

    if validation:
        file_name = f"repaired_valid_examples_{bootstrap_step}.txt"
    elif test:
        file_name = f"repaired_test_examples_{bootstrap_step}.txt"
    else:
        file_name = f"repaired_examples_{bootstrap_step}.txt"

    with open(os.path.join(args.save_dir, file_name), "w", encoding="utf-8") as f:
        # Write pass at 1 score
        f.write(f"Pass at 1: {regular_pass_at_one}%, pass with repairing: {pass_with_repair}% \n\n")

        f.write("\n" * 10)
        f.write("Incorrect codes\n")

        for index, (code, prompt, task) in enumerate(second_incorrect_code_with_prompt_and_task):
            task_id = task["problem_id"] if "problem_id" in task else task["task_id"]
            f.write(f"Task {index}, id={task_id}:\n")
            f.write(
                f"{prompt[prompt.rfind('### Task Start ###') if 'Task Start' in prompt else 0:]}{code.strip()}\n[DONE]\n")
            f.write("-" * 100 + "\n\n\n\n")

        f.write("\n" * 10)
        f.write("Correct repaired codes\n")

        for index, (code, prompt, task) in enumerate(second_correct_code_with_prompt_and_task):
            task_id = task["problem_id"] if "problem_id" in task else task["task_id"]
            f.write(f"Task {index}, id={task_id}:\n")
            f.write(
                f"{prompt[prompt.rfind('### Task Start ###') if 'Task Start' in prompt else 0:]}{code.strip()}\n[DONE]\n")
            f.write("-" * 100 + "\n\n\n\n")

    print(f"Logged examples to {args.save_dir}/{file_name}")


def validate_model(args, bootstrap_step: int, previous_model: CodeT5Model, valid_data: Dataset,
                   valid_code_prompts: MBPPCodePrompts, experiments_logging_dict: dict, debug: bool = False):
    """
    Validates the model on the validation set.

    :param args: The arguments.
    :param bootstrap_step: The current bootstrap step.
    :param previous_model: The model to use for inference.
    :param valid_code_prompts: The prompts to use for inference.
    :param experiments_logging_dict: The dictionary to log the experiments to.
    :param valid_data: The validation dataset. Only used when valid is True.
    """
    # Perform inference on code generation task.
    perform_inference(args, previous_model, bootstrap_step, valid_code_prompts, experiments_logging_dict, valid=True,
                      valid_dataset=valid_data)


def change_to_correct_code(incorrect_code_feedback_and_task: List[Tuple[str, str, MBPPTask | APPSTask]]) -> \
        List[Tuple[str, str, MBPPTask | APPSTask]]:
    """
    Changes the incorrect code to the correct code.

    :param incorrect_code_feedback_and_task: The incorrect code with their respective feedback and task.

    :return: The correct code with their respective feedback and task.
    """

    # Add the correct code to the incorrect feedback and task.
    corrected_code_with_prompt_and_task = []
    for _, prompt, task in incorrect_code_feedback_and_task:
        if "code" in task:
            task: MBPPTask
            corrected_code_with_prompt_and_task.append((reindent_code(task["code"]), prompt, task))
        else:  # APPSTask
            task: APPSTask
            corrected_code_with_prompt_and_task.append((reindent_code(json.loads(task["solutions"])[0]), prompt, task))

    return corrected_code_with_prompt_and_task


#### FINETUNING FUNCTIONS ####

def finetune_model(args, model: CodeT5Model,
                   correct_code_prompt_task: List[Tuple[str, str, MBPPTask]],
                   tokenizer: transformers.PreTrainedTokenizer,
                   valid_data: MBPPFinetuningDataset,
                   bootstrap_step: int,
                   left_truncate=False) -> CodeT5Model:
    """
    Finetunes the model on the code given the prompt of all correct code generations.

    :param args: The arguments, partially containing the fine-tuning hyperparameters.
    :param model: The model to finetune.
    :param correct_code_prompt_task: The correct code with their respective prompt and task.
    :param tokenizer: The tokenizer to use for tokenizing the data.
    :param valid_data: The validation data to use for validation.
    :param bootstrap_step: The current bootstrap step.
    :param left_truncate: Whether to left truncate the prompt to the maximum tokenization length.

    :return: The finetuned model.
    """

    # Create a dataset for the training data.
    # Changes every bootstrapping training step, so has to be reinitialized.
    train_data = PlainBootstrappingDataset(correct_code_prompt_task, tokenizer, left_truncate=left_truncate)

    model.model.train()

    training_args = transformers.TrainingArguments(
        output_dir=os.path.join(args.save_dir, str(bootstrap_step)),
        overwrite_output_dir=True,

        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=2,

        do_predict=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type='polynomial',  # TODO: Create optimizer and scheduler manually.
        # Currently uses a power of 1, which is the same as linear decay.
        # Will need to create the AdamW optimizer and use the scheduler manually and add it to the Trainer to control
        # the args.

        logging_dir=os.path.join(args.save_dir, str(bootstrap_step)),
        logging_first_step=True,
        logging_steps=args.log_freq,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else args.num_workers,

        seed=args.seed,

        local_rank=args.local_rank,
        fp16=args.fp16,

        # For saving best model
        save_total_limit=args.save_total_limit,  # Defaults to 2
        save_steps=args.save_freq,  # Defaults to 2 (must be a round multiple of eval_steps)
        save_strategy="steps",
        load_best_model_at_end=True,

        # To prevent overfitting
        warmup_ratio=0.5,
    )

    # Early stopping callback
    early_stopping_callback = transformers.EarlyStoppingCallback(early_stopping_patience=6)

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        callbacks=[early_stopping_callback],
    )

    # Start training
    trainer.train()

    best_ckpt_path = trainer.state.best_model_checkpoint
    print(f"Best checkpoint path: {best_ckpt_path}")
    trainer.save_model()

    bootstrap_dir = os.path.join(args.save_dir, str(bootstrap_step))

    # Delete all folders in star dir to remove all data hungry checkpoints.
    for folder in os.listdir(bootstrap_dir):
        cur_dir_path = os.path.join(bootstrap_dir, folder)
        if os.path.isdir(cur_dir_path):
            shutil.rmtree(cur_dir_path)

    model.model = trainer.model  # Should be the best model, because load_best_model_at_end=True

    return model


#### GENERATION FUNCTIONS ####

def generate_code_solutions(model: CodeT5Model, code_prompts: list | MBPPCodePrompts | APPSCodePrompts, batch_size=1,
                            num_beams=1, num_return_sequences=1, temperature=1.0, do_sample=False, top_p=1.0) -> List[
    Tuple[str, str, MBPPTask]]:
    """
    Generates code solutions using the model from the given code prompts.

    :param model: The model to use for generating the code solutions.
    :param code_prompts: The code prompts to generate the code solutions from.
    :param batch_size: The batch size to use.
    :param num_beams: The number of beams to use.
    :param num_return_sequences: The number of sequences that will be returned for each prompt.
    :param temperature: The temperature to use (controls randomness of generation).
    :param do_sample: Whether to sample or not, if false then the most likely token is chosen (greedy decoding).
    :param top_p: The cumulative probability of the most likely tokens to sample from. Defaults to 1.0 (no truncation of tokens from the distribution).

    :return: The generated code solutions.
    """

    generated_code_with_task = []
    tokens_to_generate = 512
    max_tokenization_length = 600

    # Turn model in evaluation mode.
    model.model.eval()

    with torch.no_grad() and torch.cuda.amp.autocast():
        for index in tqdm(range(0, len(code_prompts), batch_size), desc="Generating code from prompt"):
            if isinstance(code_prompts, list):
                batch = code_prompts[index:index + batch_size]
                batch_prompts = [prompt for _, prompt, _ in batch]

                # Duplicate the tuples in the batch list as many times as the number of return sequences.
                batch = [batch_item for batch_item in batch for _ in range(num_return_sequences)]
            else:
                batch = [code_prompts[id] for id in range(index, index + batch_size) if id < len(code_prompts)]
                batch_prompts = [prompt for prompt, _ in batch]

                # Duplicate the tuples in the batch list as many times as the number of return sequences.
                batch = [batch_item for batch_item in batch for _ in range(num_return_sequences)]

            batch_code = model.predict_different_problems(batch_prompts, max_sample_length=tokens_to_generate,
                                                          max_length=max_tokenization_length,
                                                          num_beams=num_beams,
                                                          num_return_sequences=num_return_sequences,
                                                          temperature=temperature,
                                                          do_sample=do_sample,
                                                          top_p=top_p)

            if isinstance(code_prompts, list):
                filtered_decoded_samples = [
                    (decoded_sample[:(decoded_sample.find("[DONE]") if ("[DONE]" in decoded_sample)
                                      else None)], prompt, task)
                    for decoded_sample, (_, prompt, task) in zip(batch_code, batch)]
            else:
                filtered_decoded_samples = [
                    (decoded_sample[:(decoded_sample.find("[DONE]") if ("[DONE]" in decoded_sample)
                                      else None)], prompt, task)
                    for decoded_sample, (prompt, task) in zip(batch_code, batch)]

            generated_code_with_task.extend(filtered_decoded_samples)

    return generated_code_with_task


#### VERIFICATION FUNCTIONS ####

def verify_code(code_with_prompt_and_task: List[Tuple[str, str, MBPPTask | APPSTask]],
                debug: bool = False,
                last_repair_attempt: bool = False,
                non_informative_feedback: bool = False,
                simple_feedback: bool = False,
                correct_wrong_task=False,
                valid=False,
                plain_bootstrapping=False,
                testing_function=test_mbpp_task,
                only_use_example_tests=False,
                double_check_with_hidden=False,
                return_bools=False,
                return_errors=False) -> \
        Tuple[List[Tuple[str, str, MBPPTask | APPSTask]], List[Tuple[str, str, MBPPTask | APPSTask]]] | \
        Tuple[List[Tuple[str, str, MBPPTask | APPSTask]], List[Tuple[str, str, MBPPTask | APPSTask]],
        List[bool], List[bool], List[int]] | \
        Tuple[List[Tuple[str, str, MBPPTask | APPSTask]], List[Tuple[str, str, MBPPTask | APPSTask]],
        List[bool], List[bool], List[int], List[str]]:
    """
    Verifies the code solutions by running the code and checking if it passes the tests.

    :param code_with_prompt_and_task: The code solutions to verify, with their respective prompt and task.
    :param debug: Whether to run the code or not. If False, the code will not be run and a random pass_bool will be returned.
    :param last_repair_attempt: Whether this is the last repair attempt. If True, the errors that have been successfully corrected will be saved. If False, the errors that have been encountered will be saved.
    :param non_informative_feedback: Whether to give non-informative feedback. If True, the feedback will be "Feedback: Not available."
    :param simple_feedback: Whether to give simple feedback. If True, the feedback will be "Feedback: The code above is correct." (however the model will never see this) or "Feedback: The code above is wrong. Please fix it."
    :param correct_wrong_task: Whether to add the ground truth code to the incorrect_code_feedback_and_task. If True, the ground truth code will be added.
    :param valid: Whether we are validating the model. If True, we will should not leak hidden test information.
    :param plain_bootstrapping: Whether we are doing plain bootstrapping. If True, we will not add any feedback.
    :param testing_function: The function to use for testing the code. (test_mbpp_task or test_apps_task, based on the dataset).
    :param only_use_example_tests: Whether to only use the example tests for testing the code. If True, we will only use the example tests. Only relevant for APPS tasks.
    :param double_check_with_hidden: Whether to double check the code with the hidden tests. If True, we will double check the code with the hidden tests. Only relevant for APPS tasks.
    :param return_bools: Whether to return the passing indications of the tests. If True, the pass_bools will be returned.
    :param return_errors: Whether to return the errors that have been encountered. If True, the errors will be returned.

    :return Correct code solutions (together with prompt and task), Incorrect code solutions (together with prompt and task), (Pass bools, Hidden pass bools, Problem ids, Errors)
    """

    correct_code_prompt_and_task = []
    incorrect_feedback_and_task = []
    pass_bools = []
    hidden_pass_bools = []
    problem_ids = []
    errors = []  # If last_repair_attempt is True, this will contain the errors that have been successfully
    # corrected on the last repair attempt.
    # Else (if this is the first attempt) this will contain the first error that has been encountered.

    for i, (code, prompt, task) in tqdm(enumerate(code_with_prompt_and_task), total=len(code_with_prompt_and_task),
                                        desc="Verifying code solutions"):
        code: str  # Code has already been filtered to remove the [DONE] token if it exists.
        prompt: str
        task: MBPPTask | APPSTask

        problem_ids.append(task["problem_id"] if "problem_id" in task else task["task_id"])

        # Run the file and check if it passed the tests.
        if debug:
            # 50-50 chance of passing the code.
            pass_bool = random.choice([True, False])

            if pass_bool:
                feedback_string = "Code correct, good job!"
            else:
                feedback_string = "Code wrong, please fix."
        else:
            pass_bool, feedback_string = testing_function(code, task, only_use_example_tests=only_use_example_tests)

            if double_check_with_hidden:
                # Also use the hidden bools to calculate the pass_bool.
                hidden_pass_bool, _ = test_apps_task(code, task, only_use_example_tests=False)
                hidden_pass_bools.append(hidden_pass_bool)

            if double_check_with_hidden and valid and not pass_bool:
                # Only check the hidden tests if the code did not pass the example tests. This is done because some
                # example tests are bugged such that they always evaluate to False. We do not want to repair
                # when the example test passes, this is just a shortcoming of the dataset.
                pass_bool = hidden_pass_bool

                # If pass_bool is True, then we count this as correct, even though we don't pass the example test.
                # If pass_bool is False, then we use the example test feedback to go for another repair attempt.

            if double_check_with_hidden and not valid:
                # If we are not doing validation, we are doing bootstrapping.
                # In this case, when the code passes the example tests, but not the hidden tests, we do not want to
                # give positive feedback to the model. Because we want the model to repair the code, since it is wrong.

                if pass_bool and not hidden_pass_bool:
                    feedback_string += "However, the code above is wrong. Please fix it."

                # However, if the pass_bool is True, we do reinforce on this code even though the example test says
                # it is wrong. This is because the example tests can not really be trusted.

                pass_bool = hidden_pass_bool

        if non_informative_feedback:
            feedback_string = "Feedback: Not available."

        if simple_feedback:
            if pass_bool:
                feedback_string = "Feedback: The code above is correct."
            else:
                feedback_string = "Feedback: The code above is wrong. Please fix it."

        if pass_bool:
            # The solution is correct, saving the correct code, along with prompt and task.
            correct_code_prompt_and_task.append(
                (code, prompt, task)
            )
        elif plain_bootstrapping:  # No feedback is added. Because we are using plain bootstrapping.
            incorrect_feedback_and_task.append(
                (code, prompt, task)
            )
        elif correct_wrong_task and not valid:
            # The solution is incorrect, so we will add the ground truth code.
            if "code" in task:
                correct_code_prompt_and_task.append(
                    (reindent_code(task["code"]), prompt, task)
                )
            else:
                correct_code_prompt_and_task.append(
                    (reindent_code(json.loads(task["solutions"])[0]), prompt, task)
                )
        else:
            # The solution is incorrect, saving the original prompt, along with the new prompt and task.
            if last_repair_attempt:
                prompt_to_save = prompt  # We want to finetune on this prompt as input, so we don't want extra feedback.
            else:
                # We want to do more repairs, so we will now incorporate the feedback.
                # Important to note we only use the first 1000 characters of the code, because if the code gets this long
                # it is likely not a sensible output anyway.
                prompt_to_save = prompt + code[:1000].strip() + "\n[DONE]\n\n" + feedback_string + "\n\n[ANSWER]\n"

            incorrect_feedback_and_task.append((code, prompt_to_save, task))

        pass_bools.append(pass_bool)

        if last_repair_attempt:
            # If we are doing the last repair attempt, we will save the errors that have been successfully corrected.
            if pass_bool:
                errors.append(find_error_name(prompt))
        else:
            # If we are at our first repair attempt, we will save the errors that have been encountered.
            if not pass_bool:
                errors.append(find_error_name(feedback_string))

    if return_bools:
        if return_errors:
            return correct_code_prompt_and_task, incorrect_feedback_and_task, pass_bools, hidden_pass_bools, problem_ids, errors

        return correct_code_prompt_and_task, incorrect_feedback_and_task, pass_bools, hidden_pass_bools, problem_ids

    return correct_code_prompt_and_task, incorrect_feedback_and_task


#### EXPERIMENTS FUNCTIONS ####

def perform_inference(args, previous_model, step, prompts, experiments_logging_dict: dict, valid: bool = False,
                      valid_dataset=None):
    """
    Performs inference on the code generation task.

    :param args: The arguments.
    :param previous_model: The model to use for inference.
    :param step: The current bootstrapping step.
    :param prompts: The prompts to use for inference.
    :param experiments_logging_dict: The dictionary to log the experiments to.
    :param valid: Whether we are validating the model. If True, we will should not leak hidden test information.
    :param valid_dataset: The validation dataset. Only used when valid is True.
    """

    testing_function = test_mbpp_task if args.dataset == "MBPP" else test_apps_task

    # Perform the experiments. Then we can immediately return because we won't be using the regular inference branch.
    if args.perform_experiments and valid:
        perform_experiments(args, previous_model, step, experiments_logging_dict, valid_dataset,
                            dataset_name="valid")
        return [], [], []

    if valid:
        # Perform inference on code generation task
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=args.inference_batch_size)
    else:
        # Perform inference on code generation task
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=args.inference_batch_size,
                                                                      num_beams=args.num_beams,
                                                                      num_return_sequences=args.num_return_sequences)

    correct_code_with_prompt_and_task, incorrect_code_feedback_and_task = \
        verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=args.simple_feedback,
                    correct_wrong_task=args.correct_wrong_task, valid=valid, testing_function=testing_function,
                    plain_bootstrapping=args.training_mode == "plain")

    # Calculate the pass@1
    pass_at_1 = len(correct_code_with_prompt_and_task) / len(prompts) * 100

    # If we're not doing any repairing.
    if args.training_mode == "plain":
        # Log some of the produced codes
        log_examples(args, generated_code_with_prompt_and_task, correct_code_with_prompt_and_task, step, pass_at_1,
                     validation=valid)

        corrected_first_attempt_code = change_to_correct_code(incorrect_code_feedback_and_task)

        return correct_code_with_prompt_and_task, [], corrected_first_attempt_code

    # Repairing, so we mainly care about the incorrect code, so we will tokenize from the left.
    previous_model.change_tokenizer_truncation_direction("left")

    if valid:
        # Generate new code
        second_generated_code_with_prompt_and_task = generate_code_solutions(previous_model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size)
    else:
        # Generate new code
        second_generated_code_with_prompt_and_task = generate_code_solutions(previous_model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size,
                                                                             num_beams=args.num_beams,
                                                                             num_return_sequences=args.num_return_sequences)

    # Change the tokenizer back to default truncation direction.
    previous_model.reset_tokenizer_truncation_direction()

    # Verify the secondly generated code.
    second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task = \
        verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                    simple_feedback=args.simple_feedback, testing_function=testing_function)
    second_corrected_code_with_prompt_and_task = change_to_correct_code(second_incorrect_feedback_and_task)

    # Calculate pass with repair.
    count_initial_correct = len(correct_code_with_prompt_and_task)

    pass_with_repair = (count_initial_correct + len(second_correct_code_with_prompt_and_task)) \
                       / len(prompts) * 100

    # Log some of the produced code examples
    log_examples(args, generated_code_with_prompt_and_task, correct_code_with_prompt_and_task, step, pass_at_1,
                 validation=valid)
    log_repaired_examples(args, second_incorrect_feedback_and_task,
                          second_correct_code_with_prompt_and_task, step, pass_with_repair, pass_at_1, validation=valid)

    return correct_code_with_prompt_and_task, second_correct_code_with_prompt_and_task, second_corrected_code_with_prompt_and_task


def perform_experiments(args, model: CodeT5Model, step: int, experiments_logging_dict: dict, dataset,
                        dataset_name: str = "valid"):
    """
    Performs the specified experiments on the provided model.

    :param args: The arguments of the program.
    :param model: The model to perform the experiments on.
    :param step: The current bootstrapping step.
    :param experiments_logging_dict: The dictionary to log the results to.
    :param dataset: The dataset to perform the experiments on.
    :param dataset_name: The name of the dataset to perform the experiments on.
    """

    valid = dataset_name == "valid"
    test = dataset_name == "test"

    testing_function = test_mbpp_task if args.dataset == "MBPP" else test_apps_task

    if args.training_mode == "plain":

        if args.dataset == "MBPP":
            with open("./few_shot_examples/bootstrapping_prompt_examples.txt", "r", encoding="utf-8") as f:
                few_shot_examples = f.read()

            prompts = MBPPCodePrompts(dataset, few_shot_examples=few_shot_examples)
        else:
            if test:
                with open("./data/apps/example_test_cases/test.json", "r", encoding="utf-8") as f:
                    example_test_cases = json.load(f)
            else:
                example_test_cases = None
            prompts = APPSCodePrompts(dataset, example_test_cases=example_test_cases,
                                      require_example_test_cases=test, load_valid_dataset=valid, testing=test,
                                      remove_test_examples_from_prompt=valid)

        if not args.only_perform_basic_tests:
            # 2. Experiment: Baseline 2, beam search with beam size 2 and num_returns 2. Calculate pass@2.

            # Perform inference on code generation task
            generated_code_with_prompt_and_task = generate_code_solutions(model, prompts,
                                                                          batch_size=args.beam_search_batch_size,
                                                                          num_beams=2,
                                                                          num_return_sequences=2)

            # Calculate the pass@k
            pass_at_2 = calc_pass_at_k(generated_code_with_prompt_and_task, k=2, testing_function=testing_function)

            # Add to the dictionary
            experiments_logging_dict[f"baseline_2_{dataset_name}"].append(pass_at_2)

        # 1. Experiment: Baseline 1
        generated_code_with_prompt_and_task = generate_code_solutions(model, prompts,
                                                                      batch_size=args.inference_batch_size)

        # Verify the generated code.
        correct_code_with_prompt_and_task, incorrect_code_feedback_and_task = \
            verify_code(generated_code_with_prompt_and_task, debug=args.db, plain_bootstrapping=True,
                        correct_wrong_task=args.correct_wrong_task, valid=True, testing_function=testing_function)

        # Calculate the pass@1
        pass_at_1 = len(correct_code_with_prompt_and_task) / len(prompts) * 100

        # Add to the dictionary
        experiments_logging_dict[f"baseline_1_{dataset_name}"].append(pass_at_1)

        log_examples(args, generated_code_with_prompt_and_task, correct_code_with_prompt_and_task, step, pass_at_1,
                     validation=valid, test=test)

        # 3. Experiment: Baseline 3, temperature sampling with temperature 0.8, return 2. Calculate pass@2.
        generated_code_with_prompt_and_task = generate_code_solutions(model, prompts,
                                                                      batch_size=args.inference_batch_size,
                                                                      temperature=0.8,
                                                                      do_sample=True,
                                                                      num_return_sequences=2,
                                                                      top_p=1)

        # Calculate the pass@k
        pass_at_2 = calc_pass_at_k(generated_code_with_prompt_and_task, k=2, testing_function=testing_function)

        # Add to the dictionary
        experiments_logging_dict[f"baseline_3_{dataset_name}"].append(pass_at_2)

    # 1. Simple Feedback + Repair
    if not args.only_perform_basic_tests or args.training_mode == "simple_feedback":
        if args.dataset == "MBPP":
            with open("./few_shot_examples/simple_feedback_prompt_examples.txt", "r", encoding="utf-8") as f:
                few_shot_examples = f.read()

            prompts = MBPPCodePrompts(dataset, few_shot_examples=few_shot_examples)
        else:
            if test:
                with open("./data/apps/example_test_cases/test.json", "r", encoding="utf-8") as f:
                    example_test_cases = json.load(f)
            else:
                example_test_cases = None
            prompts = APPSCodePrompts(dataset, example_test_cases=example_test_cases,
                                      remove_test_examples_from_prompt=valid,
                                      require_example_test_cases=test, load_valid_dataset=valid, testing=test)

        generated_code_with_prompt_and_task = generate_code_solutions(model, prompts,
                                                                      batch_size=args.inference_batch_size)

        # Verify the generated code.
        if args.dataset == "APPS" and test:
            # First test with example test cases only.
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, _, hidden_pass_bools, _ = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=True, valid=True,
                            testing_function=testing_function, only_use_example_tests=True,
                            double_check_with_hidden=True,
                            return_bools=True)

            # Calculate the pass@1
            pass_at_1 = sum(hidden_pass_bools) / len(prompts) * 100
        else:
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=True, valid=True,
                            testing_function=testing_function)

            # Calculate the pass@1
            pass_at_1 = len(correct_code_with_prompt_and_task) / len(prompts) * 100

        # Generate new code
        model.change_tokenizer_truncation_direction("left")
        second_generated_code_with_prompt_and_task = generate_code_solutions(model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size)
        model.reset_tokenizer_truncation_direction()

        # Verify the secondly generated code.
        second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task = \
            verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                        simple_feedback=True, testing_function=testing_function)

        # Calculate pass with repair.
        if args.dataset != "MBPP" and test:
            count_initial_correct = sum(hidden_pass_bools)
        else:
            count_initial_correct = len(correct_code_with_prompt_and_task)

        pass_with_repair = (count_initial_correct + len(second_correct_code_with_prompt_and_task)) \
                           / len(prompts) * 100

        # Add to the dictionary
        experiments_logging_dict[f"simple_feedback_{dataset_name}"].append(pass_at_1)
        experiments_logging_dict[f"simple_feedback_repair_{dataset_name}"].append(pass_with_repair)

        if args.training_mode == "simple_feedback":
            # Log some of the produced codes
            log_examples(args, generated_code_with_prompt_and_task, correct_code_with_prompt_and_task, step, pass_at_1,
                         validation=valid, test=test)
            log_repaired_examples(args, second_incorrect_feedback_and_task,
                                  second_correct_code_with_prompt_and_task, step, pass_with_repair, pass_at_1,
                                  validation=valid, test=test)

    # 2. Full Feedback + Repair
    if not args.only_perform_basic_tests or args.training_mode == "full_feedback":
        # Create the prompts
        if args.dataset == "MBPP":
            with open("./few_shot_examples/sdr_prompt_examples.txt", "r", encoding="utf-8") as f:
                few_shot_examples = f.read()

            prompts = MBPPCodePrompts(dataset, few_shot_examples=few_shot_examples)
        else:
            if test:
                with open("./data/apps/example_test_cases/test.json", "r", encoding="utf-8") as f:
                    example_test_cases = json.load(f)
            else:
                example_test_cases = None
            prompts = APPSCodePrompts(dataset, example_test_cases=example_test_cases, require_example_test_cases=test,
                                      load_valid_dataset=valid, testing=test, remove_test_examples_from_prompt=valid)
        # Generate the code
        generated_code_with_prompt_and_task = generate_code_solutions(model, prompts,
                                                                      batch_size=args.inference_batch_size)

        # Verify the generated code.
        if args.dataset == "APPS" and test:
            # First test with example test cases only.
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, _, hidden_pass_bools, _ = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, valid=True,
                            testing_function=testing_function, only_use_example_tests=True,
                            double_check_with_hidden=True, return_bools=True)

            # Calculate the pass@1
            pass_at_1 = sum(hidden_pass_bools) / len(prompts) * 100
        else:
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, valid=True,
                            testing_function=testing_function)

            pass_at_1 = len(correct_code_with_prompt_and_task) / len(prompts) * 100

        # Generate new code
        model.change_tokenizer_truncation_direction("left")
        second_generated_code_with_prompt_and_task = generate_code_solutions(model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size)
        model.reset_tokenizer_truncation_direction()

        # Verify the secondly generated code.
        second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task = \
            verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                        testing_function=testing_function)

        # Calculate pass with repair.
        if args.dataset == "APPS" and test:
            count_initial_correct = sum(hidden_pass_bools)
        else:
            count_initial_correct = len(correct_code_with_prompt_and_task)

        pass_with_repair = (count_initial_correct + len(second_correct_code_with_prompt_and_task)) \
                           / len(prompts) * 100

        if args.training_mode == "full_feedback":
            # Log some of the produced codes
            log_examples(args, generated_code_with_prompt_and_task, correct_code_with_prompt_and_task, step, pass_at_1,
                         validation=valid, test=test)
            log_repaired_examples(args, second_incorrect_feedback_and_task,
                                  second_correct_code_with_prompt_and_task, step, pass_with_repair, pass_at_1,
                                  validation=valid, test=test)

        # Add to the dictionary
        experiments_logging_dict[f"full_feedback_{dataset_name}"].append(pass_at_1)
        experiments_logging_dict[f"full_feedback_repair_{dataset_name}"].append(pass_with_repair)

    # Save the logging dictionary
    with open(os.path.join(args.save_dir, "experiments_logging_dict.json"), "w", encoding="utf-8") as f:
        json.dump(experiments_logging_dict, f, indent=4)


def load_model(args, model_path: str):
    """
    This function loads the language model and tokenizer.

    :param args: The arguments that are passed to the program.
    :param model_path: The path to the model that should be loaded.
    """

    # Load model and tokenizer
    print("Loading model and tokenizer from {}...".format(model_path))
    # Generate using the model.
    if args.model_path:
        if args.load_locally:
            original_model = CodeT5Model(model_path=args.model_path, local_files_only=True)
        else:
            original_model = CodeT5Model(model_path=args.model_path)
    else:
        original_model = CodeT5Model()
    tokenizer = original_model.tokenizer
    print('Finished loading model {}'.format(args.model_path))
    return original_model, tokenizer


def run_bootstrapping(args):
    """
    This function runs the bootstrapping algorithm.

    :param args: The arguments that are passed to the program.
    """
    model_path = args.model_path if args.model_path is not None else f"Salesforce/codet5-large-ntp-py"

    truncation_side = "left" if args.dataset == "MBPP" else "right"

    original_model, tokenizer = load_model(args, model_path)
    original_model.truncation_side = truncation_side
    original_model.reset_tokenizer_truncation_direction()

    # Load few shot examples here by reading one of the text files depending on the training mode
    if args.training_mode == "plain":
        print("Training mode is plain bootstrapping, not using feedback.")
        with open("./few_shot_examples/bootstrapping_prompt_examples.txt", "r", encoding="utf-8") as f:
            few_shot_examples = f.read()
    elif args.training_mode == "simple_feedback":
        print("Training mode is simple feedback bootstrapping.")
        with open("./few_shot_examples/simple_feedback_prompt_examples.txt", "r", encoding="utf-8") as f:
            few_shot_examples = f.read()
    else:
        print("Defaulting to the full feedback bootstrapping training mode.")
        with open("./few_shot_examples/sdr_prompt_examples.txt", "r", encoding="utf-8") as f:
            few_shot_examples = f.read()

    # Loading dataset
    print("Loading dataset...")

    if args.dataset == "MBPP":
        dataset = load_mbpp_dataset()
    else:
        dataset = load_apps_dataset()

    train_data = dataset["train"]
    valid_data = dataset["train"]  # This is not an error, we don't have a validation set for APPS.
    # This should be used to construct a validation set.
    test_data = dataset["test"]

    if args.dataset == "MBPP":  # APPS doesn't have an explicit validation set, but MBPP does.
        valid_data = dataset["validation"]

    debug_dataset_loader = load_debug_mdpp_dataset if args.dataset == "MBPP" else load_debug_apps_dataset

    if args.db:
        train_data = debug_dataset_loader(length=4, split="train")
        test_data = debug_dataset_loader(length=4, split="test")
        valid_data = train_data

        if args.dataset == "MBPP":
            valid_data = debug_dataset_loader(length=4, split="validation")
    if args.dataset_size > 0:
        train_data = debug_dataset_loader(length=args.dataset_size, split="train")
        test_data = debug_dataset_loader(length=args.dataset_size, split="test")
        valid_data = train_data

        if args.dataset == "MBPP":
            valid_data = debug_dataset_loader(length=args.dataset_size, split="validation")

    print(f"Number of tasks in the train split: {train_data}")

    if args.dataset == "MBPP":
        print(f"Number of tasks in the valid split: {valid_data}")

    print('Finished loading dataset')

    previous_model = original_model

    if args.dataset == "MBPP":
        train_prompts = MBPPCodePrompts(train_data, few_shot_examples=few_shot_examples)
        valid_prompts = MBPPCodePrompts(valid_data, few_shot_examples=few_shot_examples)
        valid_finetuning = MBPPFinetuningDataset(valid_data, deepcopy(original_model.tokenizer),
                                                 few_shot_examples=few_shot_examples)
    else:
        train_prompts = APPSCodePrompts(train_data, remove_test_examples_from_prompt=True)
        valid_prompts = APPSCodePrompts(train_data, load_valid_dataset=True, remove_test_examples_from_prompt=True)
        valid_finetuning = APPSFinetuningDataset(train_data, deepcopy(original_model.tokenizer),
                                                 load_valid_dataset=True)

    print("Length of train prompts: ", len(train_prompts))
    print("Length of valid prompts: ", len(valid_prompts))
    print("Length of valid finetuning: ", len(valid_finetuning))

    # Replay buffer for when args.replay_buffer is on, for every bootstrapping step we add all fine-tuning examples.
    replay_buffer = []

    # Create the logging dict, especially useful for logging the results of experiments.

    # If it already exists in the save directory, load it.
    experiments_logging_dict = defaultdict(lambda: [])
    if os.path.exists(os.path.join(args.save_dir, "experiments_logging_dict.json")):
        with open(os.path.join(args.save_dir, "experiments_logging_dict.json"), "r", encoding="utf-8") as f:
            experiments_logging_dict.update(json.load(f))

    max_bootstrapping_step = -1  # Makes it such that if there is no directory present, we will start at 0.
    # Get a list of directories in the saved directory to see if we should continue training
    # from a previous bootstrapping step.
    previous_bootstrapping_steps = [int(x) for x in os.listdir(args.save_dir) if x.isdigit()]

    skip_initial_validation = False

    if previous_bootstrapping_steps:
        max_bootstrapping_step = max(previous_bootstrapping_steps)

        print("Resuming training with previous model...")
        previous_model = CodeT5Model(model_path=os.path.join(args.save_dir, str(max_bootstrapping_step)),
                                     local_files_only=True)
        previous_model.truncation_side = truncation_side
        previous_model.reset_tokenizer_truncation_direction()

        tokenizer = previous_model.tokenizer

        if len(list(experiments_logging_dict.values())[0]) == max_bootstrapping_step + 2:
            skip_initial_validation = True
            print("Skipping initial validation because it has already been done. (assuming validate_first_step was on)")

    for step in range(max_bootstrapping_step + 1, args.bootstrapping_steps):
        print(f"Starting step {step}/{args.bootstrapping_steps}...")

        # Get validation results from previous model.
        if (args.validate_first_step or step > 0) and not skip_initial_validation:
            print("Getting validation results from previous model...")
            validate_model(args, step, previous_model, valid_data, valid_prompts, experiments_logging_dict,
                           debug=args.db)

        if args.perform_testing and not skip_initial_validation:
            print("Getting testing results from previous model...")
            perform_experiments(args, previous_model, step, experiments_logging_dict, test_data,
                                dataset_name="test")

        skip_initial_validation = False

        # Once we've reached the last bootstrapping step we can stop. No need to fine-tune if we are not going to
        # validate/test.
        if step == args.bootstrapping_steps - 1:
            break

        correct_code_with_prompt_and_task, second_correct_code_with_prompt_and_task, second_corrected_code_with_prompt_and_task = \
            perform_inference(args, previous_model, step, train_prompts, experiments_logging_dict)

        # Remove the previous model from memory (checkpoint should be saved somewhere)
        del original_model
        del previous_model
        torch.cuda.empty_cache()

        # Create a new model from the original model, which will be the model in the next loop.
        if args.finetune_on_original_model:
            # If we use the finetuned model during the first step, we want to finetune the original model on this after.
            original_model = CodeT5Model()
        else:
            original_model, tokenizer = load_model(args, model_path)

        original_model.truncation_side = truncation_side
        original_model.reset_tokenizer_truncation_direction()

        if args.replay_buffer and not args.correct_wrong_task:
            replay_buffer += correct_code_with_prompt_and_task + second_correct_code_with_prompt_and_task \
                             + second_corrected_code_with_prompt_and_task
            finetuning_data = replay_buffer
        elif args.replay_buffer and args.correct_wrong_task:  # Only add repair tasks to the replay buffer.
            replay_buffer += second_correct_code_with_prompt_and_task \
                             + second_corrected_code_with_prompt_and_task
            finetuning_data = replay_buffer + correct_code_with_prompt_and_task
        else:
            finetuning_data = correct_code_with_prompt_and_task + second_correct_code_with_prompt_and_task \
                              + second_corrected_code_with_prompt_and_task

        # Finetune the model on the generated code, along with the potential feedback strings.
        previous_model = finetune_model(args, original_model,
                                        finetuning_data,
                                        tokenizer, valid_finetuning, step, left_truncate=args.dataset == "MBPP")


def main(args):
    """
    Entry point for training, taken from CodeRL repo.

    :param args: training arguments from argparse.
    """
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)

    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Prevent the tokenizer parallelism error when running on Snellius.
    # Figure out if FastTokenizer is needed speed-wise.
    if not args.tokenizer_parallelism:
        print("Disabling tokenizer parallelism...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start running the bootstrapping function
    run_bootstrapping(args)


if __name__ == "__main__":
    # Parse arguments
    from configs.train_configs import *

    main(args)
