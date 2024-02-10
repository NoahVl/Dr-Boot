#!/bin/bash

#Run the finetuning script
python regular_finetuning.py --batch-size-per-replica 6 --grad-acc-steps 4 --num_workers 16 --training_mode plain --seed 24 --exp_name regular_finetune_mbpp_3 --dataset MBPP --model codet5-large-ntp-py
