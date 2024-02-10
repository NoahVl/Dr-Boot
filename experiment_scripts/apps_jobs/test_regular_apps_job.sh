#!/bin/bash

#Run the testing script
python test_best_model.py --inference_batch_size 70 --num_workers 16 --validate_first_step --training_mode full_feedback --seed 18 --exp_name test_regular_2 --beam_search_batch_size 35 --dataset APPS --model_path ./exps/codet5-large-ntp-py_bs6x4_lr5e-05_seed18_expname-regular_finetune_apps_removed_test_examples_2_1307_1343/0/ --perform_testing