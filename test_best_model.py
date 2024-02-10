"""
This file contains the code to test the best model on the validation and test set.
"""

import json
import os
import pprint
from collections import defaultdict, Counter
from typing import List, Tuple

import numpy as np

from data.apps.apps_bootstrapping_dataset import APPSCodePrompts
from data.apps.apps_dataset import load_apps_dataset, load_debug_apps_dataset, APPSTask
from data.mbpp.mbpp_dataset import load_mbpp_dataset, load_debug_mdpp_dataset, MBPPTask
from data.mbpp.mbpp_bootstrapping_dataset import MBPPCodePrompts
from models.codet5_model import CodeT5Model
from itertools import compress

# Testing functions
from test_mbpp_task import test_mbpp_task
from train_sdr import generate_code_solutions, log_examples, verify_code, calc_pass_at_k, log_repaired_examples, \
    find_error_name

# If on Windows, set the following to True.
if os.name != "nt":
    from test_apps_task import test_apps_task  # Windows does not support signals.
else:
    test_apps_task = None


def load_model(args, model_path):
    """
    Loads the model and tokenizer from the given path.

    :param args: Args from argparse.
    :param model_path: The path to the model.
    """
    # Load model and tokenizer
    print("Loading model and tokenizer from {}...".format(model_path))
    original_model = CodeT5Model(model_path=model_path, local_files_only=True)
    tokenizer = original_model.tokenizer
    print('Finished loading model {}'.format(model_path))
    return original_model, tokenizer


def chen_calc_pass_at_k(generated_code_with_prompt_and_task: list, n: int, pass_at_ks: Tuple[int],
                        testing_function=test_mbpp_task,
                        pass_bools=None,
                        hidden_pass_bools=None,
                        repair_pass_bools=None,
                        problem_ids=None,
                        errors: List[str]=None) -> dict:
    """
    Calculates the pass@k using the estimator from the "Evaluating Large Language Models Trained on Code" paper by
    Chen et. al. (2021).

    :param generated_code_with_prompt_and_task: The generated code with the prompt and task.
    :param n: The number of samples per task.
    :param pass_at_ks: The k's to calculate the pass@k for.
    :param testing_function: The function to use for testing.
    :param pass_bools: The pass bools for the generated code.
    :param hidden_pass_bools: The hidden pass bools for the generated code.
    :param repair_pass_bools: The repair pass bools for the generated code.
    :param problem_ids: The problem IDs for the generated code.
    :param errors: The errors for the generated code.

    :return: A dictionary containing the pass@k for the sampled and estimated version.
    """

    task_id_to_total_correct = defaultdict(int)

    if not errors:
        errors = []

    if generated_code_with_prompt_and_task:
        pass_bools = []

        ordered_per_n = [generated_code_with_prompt_and_task[i:i + n] for i in
                         range(0, len(generated_code_with_prompt_and_task), n)]
        total_evaluated_tasks = len(ordered_per_n)

        total_correct = []

        for index, subresults in enumerate(ordered_per_n):
            passing_solutions = 0

            for subindex, (code, prompt, task) in enumerate(subresults):
                task: MBPPTask | APPSTask

                pass_bool, feedback_string = testing_function(code, task)
                passing_solutions += pass_bool
                task_id_to_total_correct[task["task_id"] if "task_id" in task else task["problem_id"]] += pass_bool
                pass_bools.append(pass_bool)
                errors.append(find_error_name(feedback_string))

            total_correct.append(passing_solutions)
    else:
        original_pass_bools = hidden_pass_bools if hidden_pass_bools else pass_bools

        if repair_pass_bools:
            full_repair_pass_bools = []

            # Match up the two lists of hidden_pass_bools and repair_pass_bools using pass_bools.
            # Everywhere where pass_bools is True, we copy hidden_pass_bools over into the full_repair_pass_bools list.
            # Everywhere where pass_bools is False, we copy the value in repair_pass_bools over into the full_repair_pass_bools list.
            repair_pass_index = 0

            for index, pass_bool in enumerate(pass_bools):
                if pass_bool:
                    full_repair_pass_bools.append(original_pass_bools[index])
                else:
                    full_repair_pass_bools.append(repair_pass_bools[repair_pass_index])
                    repair_pass_index += 1

            total_correct = [sum(full_repair_pass_bools[i:i + n]) for i in range(0, len(full_repair_pass_bools), n)]
            total_evaluated_tasks = len(total_correct)

            task_id_to_total_correct.update(Counter(list(compress(problem_ids, full_repair_pass_bools))))

            # Make sure that the task_id_to_total_correct dict has all the problem_ids in it.
            _ = [task_id_to_total_correct[id] for id in problem_ids if id not in task_id_to_total_correct]

        else:
            # In this case, we haven't repaired yet. So we just use the hidden_pass_bools if they exist,
            # otherwise we use the pass_bools.
            total_correct = [sum(original_pass_bools[i:i + n]) for i in range(0, len(original_pass_bools), n)]
            total_evaluated_tasks = len(total_correct)

            task_id_to_total_correct.update(Counter(list(compress(problem_ids, original_pass_bools))))

            # Make sure that the task_id_to_total_correct dict has all the problem_ids in it.
            _ = [task_id_to_total_correct[id] for id in problem_ids if id not in task_id_to_total_correct]


    pass_at_k_logging_dict = {"sampled": defaultdict(list),
                              "estimated": defaultdict(list),
                              "n": n,
                              "total_correct": total_correct,
                              "task_id_to_total_correct": task_id_to_total_correct}

    if errors:
        if repair_pass_bools:
            pass_at_k_logging_dict.update({"repaired_errors_to_count": dict(Counter(errors))})
        else:
            pass_at_k_logging_dict.update({"errors_to_count": dict(Counter(errors))})

    for k in pass_at_ks:
        # First calculate the non-estimator version of pass@k. We do this by using the pass bools lists.
        if repair_pass_bools:
            pass_at_k_logging_dict["sampled"][k] = \
                sum([sampled_pass_at_k(full_repair_pass_bools[i:i + n], k)
                     for i in range(0, len(full_repair_pass_bools), n)]) / total_evaluated_tasks * 100
        else:
            original_pass_bools = hidden_pass_bools if hidden_pass_bools else pass_bools
            pass_at_k_logging_dict["sampled"][k] = \
                sum([sampled_pass_at_k(original_pass_bools[i:i + n], k)
                     for i in range(0, len(original_pass_bools), n)]) / total_evaluated_tasks * 100

        pass_at_k_logging_dict["estimated"][k] = \
            sum([pass_at_k_estimator(n, c, k) for c in total_correct]) / total_evaluated_tasks * 100

    return pass_at_k_logging_dict


def pass_at_k_estimator(n, c, k):
    """
    Taken from the: "Evaluating Large Language Models Trained on Code" paper by Chen et. al. (2021).

    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def sampled_pass_at_k(pass_bools_list: List[bool], k: int):
    """
    Calculating pass@k by selecting the first k pass bools and checking if any of them are True.

    :param pass_bools_list: list of pass bools (per task).
    :param k: k in pass@$k$
    """
    k_subsample = pass_bools_list[:k]
    return int(np.any(k_subsample))


def evaluate_best_model(args, previous_model: CodeT5Model, step: int, experiments_logging_dict: dict,
                        dataset, dataset_name: str = "valid", pass_at_ks_to_test: tuple = (1, 2, 5, 10), n: int = 10):
    """
    Used to evaluate the model on the validation or test set. Specifying the number of k's to test
    (always generates the max(k's) number of samples, to then downsample for the pass@k estimates/samples).

    :param args: The arguments from argparse.
    :param previous_model: The model to evaluate.
    :param step: The step of the model.
    :param experiments_logging_dict: The dictionary to log the experiments in.
    :param dataset: The dataset to use for evaluation.
    :param dataset_name: The name of the dataset.
    :param pass_at_ks_to_test: The k's to test the pass@k for.
    :param n: The number of samples to generate per task.
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
                with open("./data/apps/example_test_cases/train.json", "r", encoding="utf-8") as f:
                    example_test_cases = None
            prompts = APPSCodePrompts(dataset, example_test_cases=example_test_cases,
                                      require_example_test_cases=test, load_valid_dataset=valid, testing=test,
                                      remove_test_examples_from_prompt=valid)

        # if not args.only_perform_basic_tests and args.dataset != "APPS":
        #     # 2. Experiment: Baseline 2, beam search with beam size 2 and num_returns 2. Calculate pass@2.
        #
        #     # Perform inference on code generation task
        #     generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
        #                                                                   batch_size=args.beam_search_batch_size,
        #                                                                   num_beams=2,
        #                                                                   num_return_sequences=2)
        #
        #     # Calculate the pass@k
        #     pass_at_2 = calc_pass_at_k(generated_code_with_prompt_and_task, k=2, testing_function=testing_function)
        #
        #     # Add to the dictionary
        #     experiments_logging_dict[f"baseline_2_{dataset_name}"].append(pass_at_2)

        # 1. Experiment: Baseline 1
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=args.inference_batch_size)

        # Verify the generated code.
        correct_code_with_prompt_and_task, incorrect_code_feedback_and_task = \
            verify_code(generated_code_with_prompt_and_task, debug=args.db, plain_bootstrapping=True, valid=True,
                        testing_function=testing_function)

        # Calculate the pass@1
        pass_at_1 = len(correct_code_with_prompt_and_task) / len(prompts) * 100

        # Add to the dictionary
        experiments_logging_dict[f"baseline_1_{dataset_name}"].append(pass_at_1)

        log_examples(args, generated_code_with_prompt_and_task, correct_code_with_prompt_and_task, step, pass_at_1,
                     validation=valid, test=test)

        save_experiments_dict(args, experiments_logging_dict)

        # 3. Experiment: Baseline 3, temperature sampling with temperature 0.8.
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=int(
                                                                          args.inference_batch_size / n * 1.8),
                                                                      temperature=0.8,
                                                                      do_sample=True,
                                                                      num_return_sequences=n,
                                                                      top_p=1)

        # Calculate the pass@k
        pass_at_ks = chen_calc_pass_at_k(generated_code_with_prompt_and_task, n=n,
                                         pass_at_ks=pass_at_ks_to_test,
                                         testing_function=testing_function)

        # Add to the dictionary
        experiments_logging_dict[f"baseline_3_{dataset_name}"].append(pass_at_ks)

        # Save the logging dictionary
        save_experiments_dict(args, experiments_logging_dict)

    # 1. Simple Feedback + Repair
    if args.dataset != "APPS" and (not args.only_perform_basic_tests or args.training_mode == "simple_feedback"):
        if args.dataset == "MBPP":
            with open("./few_shot_examples/simple_feedback_prompt_examples.txt", "r", encoding="utf-8") as f:
                few_shot_examples = f.read()

            prompts = MBPPCodePrompts(dataset, few_shot_examples=few_shot_examples)
        else:
            if test:
                with open("./data/apps/example_test_cases/test.json", "r", encoding="utf-8") as f:
                    example_test_cases = json.load(f)
            else:
                with open("./data/apps/example_test_cases/train.json", "r", encoding="utf-8") as f:
                    example_test_cases = None
            prompts = APPSCodePrompts(dataset, example_test_cases=example_test_cases,
                                      require_example_test_cases=test, load_valid_dataset=valid, testing=test,
                                      remove_test_examples_from_prompt=valid)

        # 1. First sample greedily.
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=args.inference_batch_size)

        # Verify the generated code.
        if args.dataset == "APPS" and test:
            # First test with example test cases only.
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, hidden_pass_bools, problem_ids,\
                errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=True, valid=True,
                            testing_function=testing_function, only_use_example_tests=True,
                            double_check_with_hidden=True,
                            return_bools=True, return_errors=True)

            # Calculate the pass@1
            pre_repair_solutions = sum(hidden_pass_bools)
            pass_at_1 = pre_repair_solutions / len(prompts) * 100

        else:
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, _, problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=True, valid=True,
                            testing_function=testing_function, return_bools=True, return_errors=True)

            # Calculate the pass@1
            pre_repair_solutions = len(correct_code_with_prompt_and_task)
            pass_at_1 = pre_repair_solutions / len(prompts) * 100

        # Generate new code
        previous_model.change_tokenizer_truncation_direction("left")
        second_generated_code_with_prompt_and_task = generate_code_solutions(previous_model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size)
        previous_model.reset_tokenizer_truncation_direction()

        # Verify the secondly generated code.
        second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task, repair_bools, _, _, repaired_errors = \
            verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                        simple_feedback=True, testing_function=testing_function, return_bools=True, return_errors=True)

        # Calculate pass with repair.
        pass_with_repair = (pre_repair_solutions + len(second_correct_code_with_prompt_and_task)) / len(prompts) * 100

        # Add to the dictionary
        experiments_logging_dict[f"simple_feedback_{dataset_name}_greedy_pass@1"].append(pass_at_1)
        experiments_logging_dict[f"simple_feedback_{dataset_name}_greedy_pass@1_errors_to_count"].append(dict(Counter(errors)))

        experiments_logging_dict[f"simple_feedback_repair_{dataset_name}_greedy_repair@1"].append(pass_with_repair)
        experiments_logging_dict[f"simple_feedback_repair_{dataset_name}_greedy_repair@1_repaired_errors_to_count"].append(dict(Counter(repaired_errors)))

        # Save the logging dictionary
        save_experiments_dict(args, experiments_logging_dict)




        # 2. Then generate the n examples randomly.
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=int(
                                                                          args.inference_batch_size / n * 1.8),
                                                                      temperature=0.8,
                                                                      do_sample=True,
                                                                      num_return_sequences=n,
                                                                      top_p=1
                                                                      )

        # Verify the generated code.
        if args.dataset == "APPS" and test:
            # First test with example test cases only.
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, hidden_pass_bools, problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=True, valid=True,
                            testing_function=testing_function, only_use_example_tests=True, double_check_with_hidden=True,
                            return_bools=True, return_errors=True)

            # Calculate the pass@k's
            pass_at_ks = chen_calc_pass_at_k([], n=n, pass_at_ks=pass_at_ks_to_test,
                                             testing_function=testing_function,
                                             pass_bools=pass_bools,
                                             hidden_pass_bools=hidden_pass_bools,
                                             problem_ids=problem_ids,
                                             errors=errors
                                             )
        else:
            hidden_pass_bools = None
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, _, problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, simple_feedback=True, valid=True,
                            testing_function=testing_function, return_bools=True, return_errors=True)

            # Calculate the pass@k's
            # TODO: Make note that evaluating this way (with the estimator) means that the repair attempts might be different than the original code.
            pass_at_ks = chen_calc_pass_at_k([], n=n, pass_at_ks=pass_at_ks_to_test,
                                             testing_function=testing_function,
                                             pass_bools=pass_bools,
                                             problem_ids=problem_ids,
                                             errors=errors)

        # Generate new code
        previous_model.change_tokenizer_truncation_direction("left")
        second_generated_code_with_prompt_and_task = generate_code_solutions(previous_model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size,
                                                                             temperature=0.8,
                                                                             do_sample=True,
                                                                             )
        previous_model.reset_tokenizer_truncation_direction()

        # Verify the secondly generated code.
        second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task, repair_bools, _, _, repaired_errors = \
            verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                        simple_feedback=True, testing_function=testing_function, return_bools=True, return_errors=True)

        # Calculate pass with repair.
        pass_with_repair = chen_calc_pass_at_k([], n=n, pass_at_ks=pass_at_ks_to_test,
                                               testing_function=testing_function,
                                               pass_bools=pass_bools,
                                               hidden_pass_bools=hidden_pass_bools,
                                               repair_pass_bools=repair_bools,
                                               problem_ids=problem_ids,
                                               errors=repaired_errors)

        # Add to the dictionary
        experiments_logging_dict[f"simple_feedback_{dataset_name}"].append(pass_at_ks)
        experiments_logging_dict[f"simple_feedback_repair_{dataset_name}"].append(pass_with_repair)

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
            prompts = APPSCodePrompts(dataset, example_test_cases=example_test_cases,
                                      require_example_test_cases=test, load_valid_dataset=valid, testing=test,
                                      remove_test_examples_from_prompt=valid)

        # 1. First sample greedily.
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=args.inference_batch_size)

        # Verify the generated code.
        if args.dataset == "APPS" and test:
            # First test with example test cases only.
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, hidden_pass_bools, \
                problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, valid=True,
                            testing_function=testing_function, only_use_example_tests=True,
                            double_check_with_hidden=True, return_bools=True, return_errors=True)

            # Calculate the pass@1
            pre_repair_solutions = sum(hidden_pass_bools)
            pass_at_1 = pre_repair_solutions / len(prompts) * 100
        else:
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, _, problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, valid=True,
                            testing_function=testing_function, return_bools=True, return_errors=True)

            # Calculate the pass@1
            pre_repair_solutions = len(correct_code_with_prompt_and_task)
            pass_at_1 = pre_repair_solutions / len(prompts) * 100

        # Generate new code
        previous_model.change_tokenizer_truncation_direction("left")
        second_generated_code_with_prompt_and_task = generate_code_solutions(previous_model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size)
        previous_model.reset_tokenizer_truncation_direction()

        # Verify the secondly generated code.
        second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task, repair_bools, _, _, repaired_errors = \
            verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                        simple_feedback=True, testing_function=testing_function, return_bools=True, return_errors=True)

        # Calculate pass with repair.
        pass_with_repair = (pre_repair_solutions + len(second_correct_code_with_prompt_and_task)) / len(prompts) * 100

        # Add to the dictionary
        experiments_logging_dict[f"full_feedback_{dataset_name}_greedy_pass@1"].append(pass_at_1)
        experiments_logging_dict[f"full_feedback_{dataset_name}_greedy_pass@1_errors_to_count"].append(dict(Counter(errors)))

        experiments_logging_dict[f"full_feedback_repair_{dataset_name}_greedy_repair@1"].append(pass_with_repair)
        experiments_logging_dict[f"full_feedback_repair_{dataset_name}_greedy_repair@1_errors_to_count"].append(dict(Counter(repaired_errors)))

        # Save the logging dictionary
        save_experiments_dict(args, experiments_logging_dict)




        # 2. Then sample n programs.
        # Generate the code
        generated_code_with_prompt_and_task = generate_code_solutions(previous_model, prompts,
                                                                      batch_size=int(
                                                                          args.inference_batch_size / n * 1.8),
                                                                      temperature=0.8,
                                                                      do_sample=True,
                                                                      num_return_sequences=n,
                                                                      top_p=1
                                                                      )

        # Verify the generated code.
        if args.dataset == "APPS" and test:
            # First test with example test cases only.
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, hidden_pass_bools, problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, valid=True,
                            testing_function=testing_function, only_use_example_tests=True,
                            double_check_with_hidden=True, return_bools=True, return_errors=True)

            # Calculate the pass@k's
            pass_at_ks = chen_calc_pass_at_k(
                [], n=n, pass_at_ks=pass_at_ks_to_test,
                testing_function=testing_function,
                pass_bools=pass_bools,
                hidden_pass_bools=hidden_pass_bools,
                problem_ids=problem_ids
            )
        else:
            hidden_pass_bools = None
            correct_code_with_prompt_and_task, incorrect_code_feedback_and_task, pass_bools, _, problem_ids, errors = \
                verify_code(generated_code_with_prompt_and_task, debug=args.db, valid=True,
                            testing_function=testing_function, return_bools=True, return_errors=True)

            # Calculate the pass@k's
            pass_at_ks = chen_calc_pass_at_k([], n=n, pass_at_ks=pass_at_ks_to_test,
                                             testing_function=testing_function,
                                             pass_bools=pass_bools,
                                             problem_ids=problem_ids,
                                             errors=errors)

        # Generate new code
        previous_model.change_tokenizer_truncation_direction("left")
        second_generated_code_with_prompt_and_task = generate_code_solutions(previous_model,
                                                                             incorrect_code_feedback_and_task,
                                                                             batch_size=args.inference_batch_size,
                                                                             temperature=0.8,
                                                                             do_sample=True,
                                                                             )
        previous_model.reset_tokenizer_truncation_direction()

        # Verify the secondly generated code.
        second_correct_code_with_prompt_and_task, second_incorrect_feedback_and_task, repair_bools, _, _, repaired_errors = \
            verify_code(second_generated_code_with_prompt_and_task, debug=args.db, last_repair_attempt=True,
                        testing_function=testing_function, return_bools=True, return_errors=True)

        # Calculate pass with repair.
        pass_with_repair = chen_calc_pass_at_k([], n=n, pass_at_ks=pass_at_ks_to_test,
                                               testing_function=testing_function,
                                               pass_bools=pass_bools,
                                               hidden_pass_bools=hidden_pass_bools,
                                               repair_pass_bools=repair_bools,
                                               problem_ids=problem_ids,
                                               errors=repaired_errors)

        # Add to the dictionary
        experiments_logging_dict[f"full_feedback_{dataset_name}"].append(pass_at_ks)
        experiments_logging_dict[f"full_feedback_repair_{dataset_name}"].append(pass_with_repair)

    save_experiments_dict(args, experiments_logging_dict)


def save_experiments_dict(args, experiments_logging_dict):
    """
    Saves the experiments logging dict to a json file.

    :param args: The arguments from argparse.
    :param experiments_logging_dict: The dictionary to save.
    """
    # Save the logging dictionary
    with open(os.path.join(args.save_dir, "experiments_logging_dict.json"), "w", encoding="utf-8") as f:
        json.dump(experiments_logging_dict, f, indent=4)


def test_model(args):
    """
    Entry point for testing the models.

    :param args: The arguments from argparse.
    """

    truncation_side = "left" if args.dataset == "MBPP" else "right"

    # Load few shot examples here by reading the text file ./few_shot_examples/bootstrapping_prompt_examples.txt
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

    if args.dataset == "MBPP":  # APPS doesn't have an explicit validation set.
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

    # Create the prompts for validation
    if args.dataset == "MBPP":
        valid_prompts = MBPPCodePrompts(valid_data, few_shot_examples=few_shot_examples)
    else:
        valid_prompts = APPSCodePrompts(train_data, load_valid_dataset=True, remove_test_examples_from_prompt=True)

    print(f"Number of tasks in the valid split: {len(valid_prompts)}")

    # Create the logging dict, especially useful for logging the results of experiments.
    experiments_logging_dict = defaultdict(lambda: [])

    model_path = args.model_path if args.model_path is not None else f"Salesforce/codet5-large-ntp-py"
    model, tokenizer = load_model(args, model_path)
    model.truncation_side = truncation_side
    model.reset_tokenizer_truncation_direction()

    # Perform experiments
    if args.perform_testing:
        evaluate_best_model(args, model, 100, experiments_logging_dict, test_data, "test",
                            pass_at_ks_to_test=(1, 2, 5, 10), n=10)
    else:
        evaluate_best_model(args, model, 100, experiments_logging_dict, valid_data, "valid",
                            pass_at_ks_to_test=(1, 2, 5, 10), n=10)


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

    # Start running the testing function
    test_model(args)


if __name__ == "__main__":
    from configs.train_configs import *

    main(args)
