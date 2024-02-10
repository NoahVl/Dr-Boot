#!/bin/bash

#Run the finetuning script
python train_sdr.py --batch-size-per-replica 6 --grad-acc-steps 4 --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --training_mode full_feedback --exp_name full_feedback_bootstrap_apps_1 --perform_experiments --beam_search_batch_size 35 --dataset APPS --only_perform_basic_tests --seed 18 --validate_first_step  --model codet5-large-ntp-py
