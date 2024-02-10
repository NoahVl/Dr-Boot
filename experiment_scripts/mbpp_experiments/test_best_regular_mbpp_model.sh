#!/bin/bash

#Run the testing script
python test_best_model.py --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --seed 24 --perform_experiments --beam_search_batch_size 35 --perform_testing --dataset MBPP --model_path ./exps/codet5-large-ntp-py_bs6x4_lr5e-05_seed24_expname-regular_finetune_mbpp_3_1307_1706/0/  --training_mode plain --exp_name test_regular_bootstrap_mbpp_3