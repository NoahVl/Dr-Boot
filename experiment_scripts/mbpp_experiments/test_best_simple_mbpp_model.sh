#!/bin/bash

#Run the testing script
python test_best_model.py --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --seed 24 --perform_experiments --beam_search_batch_size 35 --perform_testing --dataset MBPP --only_perform_basic_tests --model_path ./exps/codet5-large-ntp-py_bs6x4_lr5e-05_seed24_expname-simple_feedback_bootstrap_3_1507_1157/8/  --training_mode simple_feedback --exp_name test_simple_bootstrap_mbpp_3