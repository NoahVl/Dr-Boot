#!/bin/bash

# Training mode: full_feedback, even though testing plain. This is because we don't have few shot prompts for APPS due to context size of CodeT5 so it does not matter.

#Run the testing script
python test_best_model.py --inference_batch_size 70 --num_workers 16 --validate_first_step --training_mode full_feedback --seed 42 --exp_name test_plain_feedback_1 --beam_search_batch_size 35 --dataset APPS --model_path ./exps/codet5-large-ntp-py_bs6x4_lr5e-05_seed42_expname-plain_bootstrap_apps_1_1307_1138/0/ --perform_testing