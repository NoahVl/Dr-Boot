"""
This file contains the code to test the best model on the validation and test set.
"""


import threading
from typing import Tuple, List
import numbers
import multiprocessing

from tqdm import tqdm

from data.mbpp.mbpp_dataset import MBPPTask, load_mbpp_dataset

# Task specific imports
import sys  # For task with task_id 596, it uses sys for an assert. A model should not be required to import things necessary to make the right side of an assert run.


def is_close(x, y, rel_tol=1e-9, abs_tol=0.0):
    """
    Check if two numbers are close to each other within a given tolerance.
    Written by: ChatGPT, using this rather than math.isclose to prevent using imports.

    Args:
        x (numbers.Numbers): The first number.
        y (numbers.Numbers): The second number.
        rel_tol (float): The relative tolerance. Defaults to 1e-9.
        abs_tol (float): The absolute tolerance. Defaults to 0.0.

    Returns:
        bool: True if x and y are close to each other within the given tolerances, False otherwise.
    """
    diff = abs(x - y)
    return diff <= max(rel_tol * max(abs(x), abs(y)), abs_tol)


def test_mbpp_task(code_answer: str, task: MBPPTask, include_standard_imports=False, timeout=10, **kwargs):
    """
    Tests a proposed solution to a task. Uses multiprocessing to allow for seperation of imports and allow for tiemouts on Windows.

    :param code_answer: The proposed solution to the task.
    :param task: The task to test the solution on.
    :param include_standard_imports: Whether to include the standard imports in the solution.
    :return: A tuple containing a boolean indicating whether the solution passed the tests and a string containing the
    feedback to the model.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_task_as_process_with_queue, args=(code_answer, task, include_standard_imports, result_queue))
    p.start()
    p.join(timeout=timeout)

    if p.exitcode == 0:  # If the process exited normally.
        result = result_queue.get()  # Get the result from the queue.
        return result

    if p.is_alive():  # If the process is still running (usually happens during IO blocking).
        p.terminate()  # Terminate the process.

    return False, f"Feedback: The code above timed out. Please fix it."


def run_task_as_process_with_queue(code_answer: str, task: MBPPTask, include_standard_imports: bool, result_queue: multiprocessing.Queue):
    """
    Helper function to run the task as a process with a queue to allow for timeouts with blocking IO.
    """
    # Execute the code.
    result = helper_test_mbpp_task(code_answer, task, include_standard_imports)

    # Put the result into the queue.
    result_queue.put(result)


def helper_test_mbpp_task(original_code_answer: str, task: MBPPTask, include_standard_imports=False) -> Tuple[bool, str]:
    """
    Tests a proposed solution to a task. This code should be run in a separate process, because it will execute the code_answer in the current environment.

    :param code_answer: The proposed solution to the task.
    :param task: The task to test the solution on.
    :param include_standard_imports: Whether to include the standard imports in the solution.
    :return: A tuple containing a boolean indicating whether the solution passed the tests and a string containing the
    feedback to the model.
    """
    # Dictionaries for the global and local variables during exec and eval, aids with imports.
    global_dict = {}
    local_dict = {}

    code_answer = original_code_answer

    if include_standard_imports:
        standard_imports = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        code_answer = standard_imports + original_code_answer

    # Executing the (proposed) solution, if this fails we will give back an error message.
    try:
        # Execute code such that we have the function, libraries and auxiliary classes needed for testing in our environment
        exec(code_answer, global_dict, local_dict)
        global_dict.update(local_dict)  # update the global dict with local variables
    except Exception as e:
        # Print type of error and message
        # print(type(e).__name__ + ":", e)
        feedback_str = f'''Feedback: The above code produces the following error while executing:\n"""\n{type(e).__name__ + f': {e}'}\n"""\nSo the code is not correct. Please fix it.'''
        return (False, feedback_str)

    # Get these dynamically from the task dict
    test_setup_code = task["test_setup_code"]

    # Executing the setup code for the tests, this should never fail.
    try:
        # Execute the setup code
        exec(test_setup_code, global_dict, local_dict)
        global_dict.update(local_dict)  # update the global dict with local variables
    except:
        print("Error in test setup code, this should never happen because it's part of the dataset")
        exit(1)

    inputs = []  # Contains tuples of the function in str and the corresponding target output
    try:
        for assert_str in task["test_list"]:
            # First we find the function name and the input arguments in the assert
            assert_removed = assert_str[len("assert "):].split("==")
            input = assert_removed[0].strip()
            target = eval(assert_removed[1].strip())

            # We add the input and target to the inputs list
            inputs.append((input, target))

    except Exception as e:
        # print("We failed to properly evaluate the assertions, likely meaning that the supporting code to create the objects in the assertions is missing.")
        # print(type(e).__name__ + ":", e)

        feedback_str = f'''Feedback: With the above function, {assert_str} returns the following error:\n"""\n{type(e).__name__ + f': {e}'}\n"""
So the code does not pass the assertion. Please fix it.'''

        return (False, feedback_str)

    try:
        results = []  # Contains tuples of a boolean (signifying whether the test passed) and the result of the function
        # Execute the function on these inputs
        for input, target in inputs:
            result = eval(input, global_dict, local_dict)

            if isinstance(result, numbers.Number) and isinstance(target, numbers.Number):
                passed_test = is_close(result, target)
            else:
                passed_test = result == target

            results.append((passed_test, result))

    except Exception as e:
        print("We failed to properly call the function or the function crashed or we got a timeout")

        # Consider these cases individually, make seperate debug messages for all of them and perhaps different exit codes.
        feedback_str = f'''Feedback: With the above function, {input} returns the following error:\n"""\n{type(e).__name__ + f': {e}'}\n"""
So the code does not pass the assertion. Please fix it.'''
        return (False, feedback_str)

    # Create boolean indicating whether all tests passed.
    all_tests_passed = all([result[0] for result in results])

    # Determine if the equal sign is connected with a space or not in the assert.
    connecting_equals = "==" if task["test_list"][0].find(")==") >= 0 else " == "

    # Start constructing the feedback string.
    feedback_string = f"""Feedback: With the above function, {inputs[0][0]}{connecting_equals}{repr(results[0][1])}. The assertion is "{task["test_list"][0]}"."""

    # Check if we passed the first assert.
    if results[0][0]:
        feedback_string += " So the code passes the assertion."

    if all_tests_passed:
        feedback_string += " The code above is correct."
    else:
        # If we did pass the first assert, but not all "hidden" ones.
        if results[0][0]:
            feedback_string += " However, the code above is wrong. Please fix it."

        # We did not pass the first test.
        else:
            feedback_string += " So the code does not pass the assertion. Please fix it."

    return all_tests_passed, feedback_string


if __name__ == "__main__":
    # For testing.
    dataset = load_mbpp_dataset()

    specific_task = dataset["train"][5]
    result = test_mbpp_task(specific_task["code"], specific_task)
    for item in result:
        print(item)

    # exec( specific_task["code"])
    # result = eval("radian_degree(90)")
    # print(result)