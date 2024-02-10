#!/bin/bash

#Run the finetuning script
python train_sdr.py --batch-size-per-replica 6 --grad-acc-steps 4 --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --finetune_on_original_model --validate_first_step --training_mode plain --seed 24 --exp_name plain_bootstrap_mbpp_3 --perform_experiments --beam_search_batch_size 35 --perform_testing --dataset MBPP --save_dir ./codet5-large-ntp-py_bs6x4_lr5e-05_seed24_expname-plain_bootstrap_mbpp_3_1307_2223/
