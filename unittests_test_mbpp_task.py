import itertools
from unittest import TestCase

from data.mbpp.mbpp_dataset import load_mbpp_dataset
from test_mbpp_task import test_mbpp_task


class TestMBPPTester(TestCase):
    def setUp(self):
        self.dataset = load_mbpp_dataset()

        # We're skipping some tasks because the task description and asserts don't match the provided code.
        # We will provide a reasoning for all of these
        # task_id 313: It should return all positive numbers, but only returns the first.
        # task_id 362: It should return the item with the maximum occurrences in a given list, but if there are 2 items with the same number of occurrences, it only returns one.
        # task_id 436: It should return all negative numbers, but only returns the first.
        # TODO: Make a note of this in the thesis.
        self.banned_test_ids = {313, 362, 436}

    def test_all_ground_truth_code(self):
        for index, task in enumerate(itertools.chain(*[self.dataset[partition] for partition in self.dataset.keys()])):
            if task["task_id"] in self.banned_test_ids:
                continue

            print(f"Testing task {index}")
            all_tests_passed, feedback_string = test_mbpp_task(task["code"], task)
            self.assertTrue(all_tests_passed)

    def test_one_task(self):
        task = self.dataset["train"][0]
        print(task)
        print(f"Testing task {task['task_id']}")
        all_tests_passed, feedback_string = test_mbpp_task(task["code"], task)
        print(f"All tests passed: {all_tests_passed}")
        print(f"Feedback string:\n{feedback_string}")
        self.assertTrue(all_tests_passed)

    def test_passes_only_first_task(self):
        task = self.dataset["train"][0]
        print(f"Testing task {task['task_id']}")
        code =  \
"""
def max_chain_length(arr, n): 
	return 3
"""
        all_tests_passed, feedback_string = test_mbpp_task(code, task)
        print(f"All tests passed: {all_tests_passed}")
        print(f"Feedback string:\n{feedback_string}")
        self.assertFalse(all_tests_passed)

    def test_passes_only_last_task(self):
        task = self.dataset["train"][0]
        print(f"Testing task {task['task_id']}")
        code = \
"""
def max_chain_length(arr, n): 
   return 5
"""
        all_tests_passed, feedback_string = test_mbpp_task(code, task)
        print(f"All tests passed: {all_tests_passed}")
        print(f"Feedback string:\n{feedback_string}")
        self.assertFalse(all_tests_passed)

    def test_missing_import_outside_function(self):
        task = self.dataset["train"][0]
        print(f"Testing task {task['task_id']}")
        code = \
"""
print(math.pi)

def max_chain_length(arr, n):
    return 3
"""
        all_tests_passed, feedback_string = test_mbpp_task(code, task)
        print(f"All tests passed: {all_tests_passed}")
        print(f"Feedback string:\n{feedback_string}")
        self.assertFalse(all_tests_passed)

    def test_missing_import_inside_function(self):
        task = self.dataset["train"][0]
        print(f"Testing task {task['task_id']}")
        code = \
"""
def max_chain_length(arr, n):
    print(math.pi)
    return 3
"""
        all_tests_passed, feedback_string = test_mbpp_task(code, task)
        print(f"All tests passed: {all_tests_passed}")
        print(f"Feedback string:\n{feedback_string}")
        self.assertFalse(all_tests_passed)

    def test_blocking_inside_function(self):
        task = self.dataset["train"][0]
        print(f"Testing task {task['task_id']}")
        code = \
"""
def max_chain_length(arr, n):
    while True:
        pass
    return 3
"""
        all_tests_passed, feedback_string = test_mbpp_task(code, task)
        print(f"All tests passed: {all_tests_passed}")
        print(f"Feedback string:\n{feedback_string}")
        self.assertFalse(all_tests_passed)

