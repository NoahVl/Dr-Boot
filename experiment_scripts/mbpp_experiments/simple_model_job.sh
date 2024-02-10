#!/bin/bash

#Run the finetuning script
python train_sdr.py --batch-size-per-replica 6 --grad-acc-steps 4 --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --finetune_on_original_model --validate_first_step --training_mode simple_feedback --seed 24 --exp_name simple_feedback_bootstrap_3 --perform_experiments --beam_search_batch_size 35 --perform_testing --dataset MBPP
