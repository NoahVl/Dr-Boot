"""
This file contains code for performing regular fine-tuning using the APPS or MBPP dataset.
These trained models are compared to the bootstrapped models in the thesis.
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
from data.mbpp.mbpp_finetuning_dataset import MBPPFinetuningDataset
from data.utils import reindent_code
from models.codet5_model import CodeT5Model


def finetune_model(args, model: CodeT5Model,
                   train_data,
                   valid_data: MBPPFinetuningDataset,
                   tokenizer: transformers.PreTrainedTokenizer,
                   bootstrap_step=0) -> CodeT5Model:
    """
    Finetunes the model on the given dataset.

    :param args: The hyper-parameters for training.
    :param model: The model to finetune.
    :param train_data: The training data.
    :param valid_data: The validation data.
    :param bootstrap_step: The current bootstrap step.

    :return The finetuned model.
    """

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
        lr_scheduler_type='polynomial',
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


def load_model(args, model_path):
    """
    Loads the model and tokenizer from the given path.

    :param args: The arguments for training.
    :param model_path: The path to the model.

    :return The loaded model and tokenizer.
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


def run_training(args):
    """
    Runs the bootstrapping algorithm for training.

    :param args: The arguments for training.
    """

    truncation_side = "left" if args.dataset == "MBPP" else "right"
    model_path = args.model_path if args.model_path is not None else f"Salesforce/codet5-large-ntp-py"

    original_model, tokenizer = load_model(args, model_path)
    original_model.truncation_side = truncation_side
    original_model.reset_tokenizer_truncation_direction()

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

    if args.dataset == "MBPP":
        train_finetuning = MBPPFinetuningDataset(train_data, deepcopy(original_model.tokenizer),
                                                 few_shot_examples=few_shot_examples)
        valid_finetuning = MBPPFinetuningDataset(valid_data, deepcopy(original_model.tokenizer),
                                                 few_shot_examples=few_shot_examples)
    else:
        train_finetuning = APPSFinetuningDataset(train_data, deepcopy(original_model.tokenizer),
                                                 remove_example_test_cases=True)
        valid_finetuning = APPSFinetuningDataset(train_data, deepcopy(original_model.tokenizer),
                                                 load_valid_dataset=True)

    finetune_model(args, original_model, train_finetuning, valid_finetuning, tokenizer, bootstrap_step=0)


def main(args):
    """
    Entry point for training, taken from CodeRL repo.

    :param args: Training arguments from argparse.
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
    run_training(args)


if __name__ == "__main__":
    from configs.train_configs import *

    main(args)
