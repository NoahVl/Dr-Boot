"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in https://github.com/salesforce/CodeRL/blob/main/LICENSE.txt
or https://opensource.org/licenses/BSD-3-Clause

This script is used to test APPSTasks during training and testing of the model.

Script originally from the CodeRL repo (https://github.com/salesforce/CodeRL/blob/main/test_one_solution.py),
modified to work directly with APPStasks instead of return error messages (for repairing).

This script is designed for Linux as it uses signals, and will therefore not work natively on Windows.
If you want to use it on Windows then you should use it through WSL.
"""

import faulthandler
import gc
import json
import multiprocessing
import pprint
import signal
import sys
from copy import deepcopy
from datetime import datetime
from enum import Enum
from io import StringIO
from unittest.mock import patch, mock_open

import numpy as np
from pyext import RuntimeModule

from tqdm import tqdm

from data.apps.apps_bootstrapping_dataset import APPSCodePrompts
from data.apps.apps_dataset import APPSTask, load_apps_dataset, load_debug_apps_dataset

# Timeout when creating a process.
TIMEOUT = 60


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # TODO: the below line was originally commented
    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def test_apps_task(code_answer: str, task: APPSTask, include_standard_imports=True, only_use_example_tests=False):
    """
    Tests a proposed solution to a task. Uses multiprocessing to allow for seperation of imports and allow for tiemouts on Windows.

    :param code_answer: The proposed solution to the task.
    :param task: The task to test the solution on.
    :param include_standard_imports: Whether to include the standard imports in the solution.
    :return: A tuple containing a boolean indicating whether the solution passed the tests and a string containing the
    feedback to the model.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_task_as_process_with_queue, args=(
    code_answer, task, include_standard_imports, only_use_example_tests, result_queue))
    p.start()
    p.join(timeout=TIMEOUT)

    if p.exitcode == 0:  # If the process exited normally.
        result = result_queue.get()  # Get the result from the queue.
        return result

    if p.is_alive():  # If the process is still running (usually happens during IO blocking).
        p.terminate()  # Terminate the process.

    return False, f"Feedback: TimeoutError: The code above timed out. Please fix it."


def run_task_as_process_with_queue(code_answer: str, task: APPSTask, include_standard_imports: bool,
                                   only_use_example_tests: bool, result_queue: multiprocessing.Queue):
    # Execute the code against the desired input output examples.
    run_information = helper_test_apps_task(task, code_answer, use_pre_imports=include_standard_imports,
                                            only_use_example_tests=only_use_example_tests)

    results, errors, outputs, _ = run_information
    pass_bool = all(item == True for item in results)

    error = errors[0]
    feedback_string = ""

    if error:
        # "Compile" or "runtime" error
        if results[0] == -2 or results[0] == -1:
            feedback_string = f'''Feedback: The above code produces the following error while executing:\n"""\n{type(error).__name__ + f': {error}'}\n"""\nSo the code is not correct. Please fix it.'''
    if not feedback_string:
        if pass_bool:
            feedback_string = "Feedback: The code passes the test."
        else:
            feedback_string = "Feedback: OutputMismatchError: The code does not pass the test. Please fix it."

    # Put the result into the queue.
    result_queue.put((pass_bool, feedback_string))


def helper_test_apps_task(task: APPSTask, code: str, debug=False, use_pre_imports=False, only_use_example_tests=False):
    """
    Adapted from CodeRL, modified to work with pure APPSTasks, hopefully.

    How to read results:
    [-3] = no tests,
    [...,-2,...] = compile error,
    [...,-1,...] = runtime error
    [...,False,...] = failed test case
    [...,True,...] = passed test case
    """

    # 1. Check if there are tests
    if only_use_example_tests:
        try:
            in_outs = task["example_test_case"]
        except Exception as e:
            # Problem loading the tests...
            return [-3], [e], [None], code
    else:
        try:
            in_outs = json.loads(task["input_output"])
        except Exception as e:
            # Problem loading the tests...
            return [-3], [e], [None], code

    # 2. Check if there is a function name
    if in_outs.get("fn_name") is None:
        which_type = CODE_TYPE.standard_input  # Standard input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based  # Call-based
        method_name = in_outs["fn_name"]

    results = []
    errors = []
    outputs = []

    if use_pre_imports:
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
    else:
        sol = ""

    if which_type == CODE_TYPE.call_based:
        sol += code
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            if "class Solution" not in code:
                tmp = tmp_sol
            else:
                tmp = tmp_sol.Solution()
            signal.alarm(0)

        except Exception as e:
            signal.alarm(0)
            if debug:
                print(f"type 0 compilation error = {e}")
            results.append(-2)
            errors.append(e)
            outputs.append(None)
            return results, errors, outputs, sol
        signal.alarm(0)

    elif which_type == CODE_TYPE.standard_input:
        # sol
        tmp_test = code.split("\n")

        new_test = []
        for x in tmp_test:
            if (not x.startswith("from ")) and (not x.startswith("import ")):
                new_test.append("\t" + x + "\n")
            else:
                new_test.append(x + "\n")
        tmp_test = new_test

        new_test = ""
        started = False
        for i in tmp_test:
            if i.startswith("\t") and not started:
                new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                new_test += "def code():\n"
                new_test += i
                started = True
            elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                new_test += "\t" + i
            else:
                new_test += i
        tmp_test = new_test

        sol += tmp_test
        if debug:
            print(f"sol = {sol}")
            # print(f"{o}")
        method_name = "code"
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            tmp = tmp_sol
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            if debug:
                print(f"type 1 compilation error = {e}")
            results.append(-2)
            errors.append(e)
            outputs.append(None)
            return results, errors, outputs, sol
        signal.alarm(0)
    if debug:
        print(f"get method = {datetime.now().time()}")

    try:
        method = getattr(tmp, method_name.strip())  # get_attr second arg must be str
    except:
        signal.alarm(0)
        e = sys.exc_info()
        print(f"unable to get function error = {e}")
        return results + [False], errors + [e], outputs, sol

    # for index, inputs in enumerate(in_outs["inputs"]):
    for index, inputs in tqdm(enumerate(in_outs["inputs"]), total=len(in_outs["inputs"]), ncols=0, leave=False):

        gc.collect()

        # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
        try:
            if isinstance(inputs[0], dict):
                inputs = [{int(k): v for k, v in inputs[0].items()}]
        except:
            True
        try:
            if isinstance(in_outs["outputs"][index], dict):
                in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
        except:
            True
        try:
            if isinstance(in_outs["outputs"][index][0], dict):
                in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
        except:
            True

        if debug:
            print(
                f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")
        if which_type == CODE_TYPE.call_based:  # Call-based
            signal.alarm(timeout)
            faulthandler.enable()
            try:
                # print("------------")
                # print(inputs)
                output = method(*inputs)

                # ground truth sequences are not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = output == in_outs["outputs"][index]
                if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                    tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                # ground truth sequences are not tuples
                try:
                    if isinstance(output[0], tuple):
                        tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                except:
                    True
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)

                # reset the alarm
                signal.alarm(0)

            except Exception as e:
                signal.alarm(0)
                faulthandler.disable()
                if debug:
                    print(f"Standard input runtime error or time limit exceeded error = {e}")
                results.append(-1)
                errors.append(e)
                outputs.append(None)

                ## TESTING TRICK: exit loop if not pass a test case
                return results, errors, outputs, sol
                # continue
            faulthandler.disable()
            signal.alarm(0)
            if debug:
                print(
                    f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

        elif which_type == CODE_TYPE.standard_input:  # Standard input
            faulthandler.enable()
            signal.alarm(timeout)
            passed = False

            if isinstance(inputs, list):
                inputs = "\n".join(inputs)
            if isinstance(in_outs['outputs'][index], list):
                in_outs['outputs'][index] = "\n".join(in_outs['outputs'][index])

            with Capturing() as output:
                try:
                    call_method(method, inputs)
                    # reset the alarm
                    signal.alarm(0)
                    passed = True
                except Exception as e:
                    # runtime error or took too long
                    signal.alarm(0)
                    if debug:
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                    results.append(-1)
                    errors.append(e)
                    outputs.append(None)
                    ## TESTING TRICK: exit loop if not pass a test case
                    return results, errors, outputs, sol

                signal.alarm(0)

            if not passed:
                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                    else:
                        pprint.pprint(
                            f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                continue

            if passed and debug:
                print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

            if custom_compare_(output, in_outs['outputs'][index]):
                tmp_result = True
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # ground truth sequences are expressed as lists not tuples
            if isinstance(output, tuple):
                output = list(output)

            tmp_result = False
            try:
                tmp_result = (output == [in_outs["outputs"][index]])
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
                    if isinstance(output[0], str):
                        tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
            except Exception as e:
                if debug:
                    print(f"Failed check1 exception = {e}")
                pass

            if tmp_result == True:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # try one more time without \n
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = i.split("\n")
                    in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index]
                                                            if x]
            else:
                in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

            try:
                tmp_result = (output == [in_outs["outputs"][index]])
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
            except Exception as e:
                if debug:
                    print(f"Failed check2 exception = {e}")
                pass

            if tmp_result == True:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # try by converting the output into a split up list too
            if isinstance(output, list):
                output = list(filter(len, output))

            if debug:
                nl = "\n"
                if not isinstance(inputs, list):
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                else:
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

            if tmp_result == True:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            try:
                tmp_result = (output == [in_outs["outputs"][index]])
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
            except Exception as e:
                if debug:
                    print(f"Failed check3 exception = {e}")
                pass

            output_float = None

            try:
                output_float = [float(e) for e in output]
                gt_float = [float(e) for e in in_outs['outputs'][index]]
                tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
            except Exception as e:
                pass
            try:
                if isinstance(output[0], list):
                    output_float = [float(e) for e in output[0]]
                    gt_float = [float(e) for e in in_outs['outputs'][index][0]]
                    tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
            except Exception as e:
                pass

            if tmp_result == True:
                results.append(tmp_result)
                errors.append(None)
                if output_float:
                    outputs.append(output_float)
                continue

            # try by converting the stuff into split up list
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = set(i.split())
            else:
                in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

            try:
                tmp_result = (output == in_outs["outputs"][index])
            except Exception as e:
                if debug:
                    print(f"Failed check4 exception = {e}")
                continue

            if tmp_result == True:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

                # try by converting the output into a split up list too
            if isinstance(output, list):
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = i.split()
                output = list(filter(len, output))
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = set(i)
            else:
                output = output.split()
                output = list(filter(len, output))
                output = set(output)

            try:
                tmp_result = (set(frozenset(s) for s in output) == set(
                    frozenset(s) for s in in_outs["outputs"][index]))
            except Exception as e:
                if debug:
                    print(f"Failed check5 exception = {e}")

            # if they are all numbers, round so that similar numbers are treated as identical
            try:
                tmp_result = tmp_result or (set(frozenset(round(float(t), 3) for t in s) for s in output) == \
                                            set(frozenset(round(float(t), 3) for t in s) for s in
                                                in_outs["outputs"][index]))
            except Exception as e:
                if debug: print(f"Failed check6 exception = {e}")

            if tmp_result == True and debug:
                print("PASSED")

            results.append(tmp_result)
            errors.append(None)
            outputs.append(output)

            if tmp_result != True:
                ## TESTING TRICK: exit loop if not pass a test case
                return results, errors, outputs, sol

            if debug:
                nl = "\n"
                if not isinstance(inputs, list):
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                else:
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

    return results, errors, outputs, sol


# def passes_tests(task: APPSTask, code_solution: str, debug: bool = False):
#     p = multiprocessing.Process(target=run_task_as_process, args=(task, code_solution, debug))
#     p.start()
#     p.join(timeout=60)  # TODO: Make this a variable
#
#     if p.exitcode == 0:  # If the process exited normally.
#         return True
#     if p.is_alive():  # If the process is still running (usually happens during IO blocking).
#         p.terminate()  # Terminate the process.
#
#     return False


def test_ground_truth_solutions():
    passing_tasks = []
    no_tests_counter = 0
    pass_count = 0
    # Test if the verify script works for all input
    for _, task in tqdm(used_dataset, total=len(used_dataset)):
        # run_information = helper_test_apps_task(task, json.loads(task["solutions"])[0], only_use_example_tests=True,
        #                             use_pre_imports=True)
        pass_bool, feedback_str = test_apps_task(json.loads(task["solutions"])[0], task, only_use_example_tests=True,
                                                 include_standard_imports=True)

        # Count the number of tasks with no input output and therefore no tests.
        # if results == [-3]:
        #     no_tests_counter += 1
        #
        # pass_bool = all(item == True for item in results)

        pass_count += pass_bool

        if pass_bool:
            passing_tasks.append(task)

        print(feedback_str)
        print()
    pass_at_one = pass_count / len(used_dataset) * 100
    # Format pass_at_one to 2 decimal places
    print(f"{pass_at_one:.2f}% pass@1.")
    # Print the number of tasks with no tests
    print(f"{no_tests_counter}/{len(used_dataset)} tasks with no tests.")


if __name__ == "__main__":
    """
    Used only for testing.
    """
    train_dataset = load_apps_dataset()["train"]

    # Import test examples and see if they're processed correctly
    with open("./data/apps/example_test_cases/train.json", "r", encoding="utf-8") as f:
        example_test_cases = json.load(f)

    used_dataset = APPSCodePrompts(train_dataset, "", example_test_cases=example_test_cases,
                                   require_example_test_cases=True)
    valid_dataset = APPSCodePrompts(train_dataset, "", example_test_cases=example_test_cases,
                                    require_example_test_cases=True, load_valid_dataset=True)

    equivalent_tests = 0

    for _, task in tqdm(used_dataset, total=len(used_dataset)):
        task: APPSTask
        try:
            json.loads(task['input_output'])
        except:
            continue

        if json.loads(task['input_output']) == task["example_test_case"]:
            equivalent_tests += 1
        elif len(json.loads(task['input_output'])["inputs"]) < 3:
            print("Question:", task["problem_id"])
            print("Hidden test:", json.loads(task['input_output']))
            print("Extracted example test:", task["example_test_case"])
        print()

    print(f"{equivalent_tests}/{len(used_dataset)} tasks with equivalent tests.")

    # test_ground_truth_solutions()

#     example_task = [task for _, task in used_dataset if "Given a string containing only three types of characters" in task["question"]][0]
#     print(example_task["question"])
#     print(f"Ground truth solution: {json.loads(example_task['solutions'])[0]}")
#
#     # Import slow code from debugging/slow_code.py
#     supposedly_wrong_sol = \
# """
# class Solution:
# 	def checkValidString(self, s: str) -> bool:
# 		if not s:
# 			return True
# 		if s[0] == '(':
# 			if s[-1] == ')':
# 				return True
# 			else:
# 				return False
# 		if s[-1] == '(':
# 			if s[0] == ')':
# 				return True
# 			else:
# 				return False
# 		if s[0] == '*':
# 			if s[-1] == ')':
# 				return True
# 			else:
# 				return False
# 		if s[-1] == '*':
# 			if s[0] == '(':
# 				return True
# 			else:
# 				return False
# 		return True
# """
#     print(supposedly_wrong_sol)
#     pass_bool, feedback_str = test_apps_task(supposedly_wrong_sol, example_task, only_use_example_tests=True,
#                                              include_standard_imports=True)
#     print("\n\n\nOn example tests:")
#     print(feedback_str)
#     print(f"pass_bool = {pass_bool}")
#
#     # Test if it does pass the hidden tests, this would mean the verify_code code in train_sdr is not correct
#     pass_bool, feedback_str = test_apps_task(supposedly_wrong_sol, example_task, only_use_example_tests=False,
#                                              include_standard_imports=True)
#     print("\n\n\nOn hidden tests:")
#     print(feedback_str)
#     print(f"pass_bool = {pass_bool}")
#
#     # Now test the code it managed to repair.
#     supposedly_correct_sol = \
# """
# class Solution:
# 	def checkValidString(self, s: str) -> bool:
# 		if not s:
# 			return True
# 		if s[0] == '(':
# 			if s[-1] == ')':
# 				return True
# 			else:
# 				return False
# 		if s[-1] == '(':
# 			if s[0] == ')':
# 				return True
# 			else:
# 				return False
# 		if s[0] == '*':
# 			if s[-1] == ')':
# 				return True
# 			else:
# 				return False
# 		if s[-1] == '*':
# 			if s[0] == ')':
# 				return True
# 			else:
# 				return False
# 		return True
# """
#     print(supposedly_correct_sol)
#     pass_bool, feedback_str = test_apps_task(supposedly_correct_sol, example_task, only_use_example_tests=True,
#                                                 include_standard_imports=True)
#     print("\n\n\nOn example tests:")
#     print(feedback_str)
#     print(f"pass_bool = {pass_bool}")
#
#     # Test if it does pass the hidden tests, this would mean the verify_code code in train_sdr is not correct
#     pass_bool, feedback_str = test_apps_task(supposedly_correct_sol, example_task, only_use_example_tests=False,
#                                                 include_standard_imports=True)
#     print("\n\n\nOn hidden tests:")
#     print(feedback_str)
#     print(f"Actual hidden test: {json.loads(example_task['input_output'])}")
#     print(f"pass_bool = {pass_bool}")

#     example_task = [task for _, task in valid_dataset if
#                     "Bohan loves milk tea so much and he drinks one cup of milk tea every day." in task[
#                         "question"]][0]
#     print(example_task["question"])
#     print(f"Ground truth solution: {json.loads(example_task['solutions'])[0]}")
#
#     # Import slow code from debugging/slow_code.py
#     supposedly_correct_sol = \
#         """
# # cook your dish here
# for _ in range(int(input())):
#     s=input()
#     if s=='MLM':
#         print(10)
#     elif s=='MMLLMMLL':
#         print(24)
#     elif s=='MMMMMMML':
#         print(22)
#         """
#     print(supposedly_correct_sol)
#     pass_bool, feedback_str = test_apps_task(supposedly_correct_sol, example_task, only_use_example_tests=True,
#                                              include_standard_imports=True)
#     print("\n\n\nOn example tests:")
#     print(feedback_str)
#     print(f"pass_bool = {pass_bool}")
#
#     # Test if it does pass the hidden tests, this would mean the verify_code code in train_sdr is not correct
#     pass_bool, feedback_str = test_apps_task(supposedly_correct_sol, example_task, only_use_example_tests=False,
#                                              include_standard_imports=True)
#     print("\n\n\nOn hidden tests:")
#     print(feedback_str)
#     print(f"Actual hidden test: {json.loads(example_task['input_output'])}")
#     print(f"pass_bool = {pass_bool}")
