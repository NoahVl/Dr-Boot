"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in https://github.com/salesforce/CodeRL/blob/main/LICENSE.txt
or https://opensource.org/licenses/BSD-3-Clause

Script originally from the CodeRL repo (https://github.com/salesforce/CodeRL/blob/main/configs/train_configs.py),
modified to work with my hyper-parameters and added new arguments for the bootstrapping algorithm.
"""

import argparse
import random
from datetime import datetime

import transformers

parser = argparse.ArgumentParser(description="Training a language model for code generation")

# Model specific
parser.add_argument('--model', default='codet5-small', type=str, help='type of transformers model as model backbone')
parser.add_argument('--model_path', default=None, type=str, help='path to model backbone pretrained weights')
parser.add_argument('--save_dir', default=None, type=str, help='path to save trained critic model checkpoints')



# Dataloading
parser.add_argument('--train-path', default='data/APPS/train/', type=str, help='path to training data')



# Training
parser.add_argument('--epochs', default=10, type=int, help='total number of training epochs')
parser.add_argument('--lr', default=5e-5, type=float, help='training learning rate')
parser.add_argument('--batch-size-per-replica', default=4, type=int, help='batch size per GPU')
parser.add_argument('--grad-acc-steps', default=1, type=int,
                    help='number of training steps before each gradient update')
parser.add_argument('--deepspeed', default=None, type=str,
                    help='path to deepspeed configuration file; set None if not using deepspeed')
parser.add_argument('--fp16', default=True, action='store_true', help='set 16-bit training to reduce memory usage')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--db', default=False, action='store_true',
                    help='set to turn on debug mode, this might entail using dummy small data split')
parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
parser.add_argument('--num_workers', default=4, type=int, help='number of data workers')



# Logging
parser.add_argument('--log-freq', default=1, type=int, help='save training log after this number of training steps')
parser.add_argument('--save-freq', default=2, type=int,
                    help='save model checkpoints after this number of training steps')
parser.add_argument('--save_total_limit', default=2, type=int,
                    help='total of number checkpoints to keep; only keep the latest ones')


# Bootstrapping training specific (might need to be moved to a separate config file)
parser.add_argument('--bootstrapping_steps', default=10, type=int, help='Number of steps to run the bootstrapping algorithm')
parser.add_argument('--tokenizer_parallelism', default=False, action='store_true',
                    help='set to turn on tokenizer parallelism')  # Prevent the tokenizer parallelism error when running on Snellius.
parser.add_argument('--inference_batch_size', default=4, type=int,
                    help='batch size for rationale and rationalization generation, can be much higher than train batch')
                    # Batch size for inference
parser.add_argument("--compile_model", default=False, action='store_true',
                    help="set to compile the model for inference and training")
parser.add_argument("--few_shot_examples_txt_file", default="rationale_from_prompt_description_simplified_v1.txt", type=str,
                    help="Text file name containing the few shot examples, in the few_shot_examples folder")
parser.add_argument("--validate_first_step", default=False, action='store_true',
                    help="Set to validate the model before starting the bootstrapping algorithm")
parser.add_argument("--finetune_on_original_model", default=False, action='store_true',
                    help="Set to finetune on the original model while performing the bootstrapping algorithm")
parser.add_argument("--dataset_size", default=-1, type=int,
                    help="Number of examples to use from the dataset,"
                         " if specified to be bigger than 0 we use this constant.")
parser.add_argument("--num_beams", default=1, type=int,
                    help="Number of beams to use for generation")
parser.add_argument("--plain_bootstrapping", default=False, action="store_true",
                    help="Set when using plain bootstrapping, for different prompts for example.")


# Inference specific
# Repair attempts
parser.add_argument("--repair_attempts", default=1, type=int,
                    help="Number of attempts to repair the model before giving up")


# parser.add_argument("--shuffle_prompt_examples", default=False, action='store_true',
#                     help="Will shuffle the prompt examples before appending them to the task only in supported files")
# parser.add_argument("--inference_runs", default=10, type=int,
#                     help="Number of inference runs to calculate the spread over")

# parser.add_argument("--exp_name", default=str(random.randint(0, 1000)), type=str, help="Name of the experiment")
parser.add_argument("--exp_name", default=str(random.randint(0, 1000)), type=str, help="Name of the experiment")
parser.add_argument("--non_informative_feedback", default=False, action="store_true",
                    help="Set to use non informative feedback")
parser.add_argument("--simple_feedback", default=False, action="store_true",
                    help="Set to use simple feedback. Only inform if we passed the task or not. No feedback on assert.")
parser.add_argument("--correct_wrong_task", default=False, action="store_true",
                    help="Also include first pass when incorrect, "
                         "with ground truth corrected code in the training data.")
# args.num_return_sequences
parser.add_argument("--num_return_sequences", default=1, type=int,
                    help="Number of return sequences for generation, needs to be lower or equal to num_beams.")
# load_locally
parser.add_argument("--load_locally", default=False, action="store_true",
                    help="Set to load the model locally, instead of HuggingFace's site.")
# args.replay_buffer
parser.add_argument("--replay_buffer", default=False, action="store_true",
                    help="Set to use a replay buffer to store the generated examples.")

# Specifically for the experiments
parser.add_argument("--perform_experiments", default=False, action="store_true",
                    help="Perform the experiments that have been described in the thesis. These experiments refer to "
                         "the validation rounds between bootstrapping steps.")
parser.add_argument("--beam_search_batch_size", default=1, type=int,
                    help="Batch size for beam search, likely needs to be lower than inference batch size.")


# Create an argument that decides how the model is trained, the options are: plain, simple_feedback, full_feedback
parser.add_argument("--training_mode", default="full_feedback", type=str,
                    help="The way we train the model. Options are: plain, simple_feedback, full_feedback.")

# Dataset
parser.add_argument("--dataset", default="MBPP", type=str,
                    help="Dataset to use for training and testing. Options are: MBPP and APPS.")

# perform_testing
parser.add_argument("--perform_testing", default=False, action="store_true",
                    help="Set to perform testing on the test set of the model.")

# only_perform_basic_tests
parser.add_argument("--only_perform_basic_tests", default=False, action="store_true",
                    help="Set to only perform the basic tests on the model. For plain models this is baseline 2. "
                         "For the repair models this is their respective repairing method.")

# Testing specific
# Trained dir
parser.add_argument("--trained_dir", default=None, type=str,
                    help="Directory where the trained bootstrapped models are located.")

args = parser.parse_args()

# Making sure that model training is reproducible.
# TODO: Fix this, doesn't produce reproducible results for some reason...
# transformers.set_seed(args.seed)
# transformers.enable_full_determinism(args.seed)

if args.save_dir is None:
    args.save_dir = '{}_bs{}x{}_lr{}_seed{}_expname-{}'.format(
        args.model,args.batch_size_per_replica, args.grad_acc_steps, args.lr, args.seed, args.exp_name
    )

    # Also add date and time in a compact format
    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")
    args.save_dir = args.save_dir + '_' + dt_string

if args.db:
    args.save_dir = 'exps/test/{}'.format(args.save_dir)
else:
    args.save_dir = 'exps/{}'.format(args.save_dir)