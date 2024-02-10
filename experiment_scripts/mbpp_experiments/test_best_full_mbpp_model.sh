#!/bin/bash

#Run the testing script
python test_best_model.py --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --seed 24 --perform_experiments --beam_search_batch_size 35 --perform_testing --dataset MBPP --model_path ./exps/codet5-large-ntp-py_bs6x4_lr5e-05_seed24_expname-full_feedback_mbpp_bootstrap_3_1307_2223/2/  --training_mode full_feedback --exp_name test_full_bootstrap_mbpp_3