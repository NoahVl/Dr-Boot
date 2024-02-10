#!/bin/bash

#Run the testing script
python test_best_model.py --inference_batch_size 70 --num_workers 16 --training_mode full_feedback --seed 18 --exp_name proper_valid_validate_regular_finetuning_1 --beam_search_batch_size 35 --dataset APPS --model_path ./exps/codet5-large-ntp-py_bs6x4_lr5e-05_seed42_expname-regular_finetune_apps_proper_valid_0607_1638/0/