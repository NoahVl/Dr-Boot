"""
Taken from the CodeRL repository: https://github.com/salesforce/CodeRL

Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in https://github.com/salesforce/CodeRL/blob/main/LICENSE.txt
or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import json

from data.reindent import run as run_reindent

def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()


if __name__ == "__main__":
    from data.apps.apps_dataset import load_apps_dataset

    dataset = load_apps_dataset()
    print(dataset)
    print(dataset["train"][0].keys())

    first_example = dataset["train"][0]
    first_code_solution = json.loads(first_example["solutions"])[0]

    print("Original code:")
    print(first_code_solution, end="\n\n")

    print("Reindented code:")
    print(reindent_code(first_code_solution))

    print(first_code_solution == reindent_code(first_code_solution))




