#!/bin/bash

#Run the finetuning script
python regular_finetuning.py --batch-size-per-replica 6 --grad-acc-steps 4 --num_workers 16 --training_mode plain --seed 18 --exp_name regular_finetune_apps_removed_test_examples_2 --dataset APPS --model codet5-large-ntp-py