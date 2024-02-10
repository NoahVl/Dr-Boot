#!/bin/bash

#Run the finetuning script
python train_sdr.py --batch-size-per-replica 6 --grad-acc-steps 4 --inference_batch_size 70 --num_workers 16 --training_mode plain --seed 18 --exp_name plain_bootstrap_apps_2 --perform_experiments --beam_search_batch_size 35 --dataset APPS --only_perform_basic_tests --model codet5-large-ntp-py --validate_first_step --save_dir ./codet5-large-ntp-py_bs6x4_lr5e-05_seed18_expname-plain_bootstrap_apps_2_1307_1401/