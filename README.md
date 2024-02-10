# Dr. Boot: Bootstrapping Program Synthesis Language Models to Perform Repairing
Repo accompanying my [master's thesis](https://scripties.uba.uva.nl/search?id=record_54126).

<img alt="Created by DALLE-3: Woman holding a device to bootstrap an unsuspecting man, reading a poorly written title of the master&#39;s thesis." height="500" src="https://th.bing.com/th/id/OIG1.3PI.fUVVbpzSxIXVqf1K?pid=ImgGn" title="Image for repo" width="500"/>

(Image created by DALLE-3)

## Abstract
Language models for program synthesis are usually trained and evaluated on programming competition datasets (MBPP, APPS). However, these datasets are limited in size and quality, while these language models are extremely data hungry. Additionally, the language models have a misaligned program synthesis process compared to humans. While humans iteratively develop code with the help of a compiler, most program synthesis models currently produce code in one go. To solve these issues, we introduce a bootstrapping algorithm for program synthesis, that supports teaching models how to repair. We show that bootstrapping consistently outperforms regular fine-tuning. Compared to other work, our bootstrapped model performs on par with fine-tuned models that are 68% larger. Notably, bootstrapping with repairing also improves non-repairing performance compared to regular bootstrapping during inference. However, on our models, repairing during inference is likely inferior to simply sampling the same number of solutions. Furthermore, we find that there are issues with the example test cases in the training portion of the APPS dataset that are valuable to the community, as many repairing and reinforcement learning methods rely on them.

## Citation
If you use this work, please cite it as:
```
@masterthesis{vdvleuten2023,
    title        = {Dr. Boot: Bootstrapping Program Synthesis Language Models to Perform Repairing},
    author       = {Noah van der Vleuten},
    year         = 2023,
    month        = {July},
    note         = {Available at \url{https://scripties.uba.uva.nl/search?id=record_54126}},
    school       = {University of Amsterdam},
    type         = {Master's thesis}
}
```

## Repository Structure
- `configs/`: Contains the configuration file for the experiments.
- `data/`: Contains the datasets used in the thesis.
- `models/`: Contains code for running and training the CodeT5.
- `results/`: Contains the results of the experiments and analysis tools used in the thesis.
- `few_shot_examples/`: Contains the few-shot examples used in the thesis.
- `experiment_scripts/`: Contains the scripts used to run the experiments.
- `./`: Contains the training scripts with helper functions, includes the code for the bootstrapping algorithm (`train_sdr.py`).

## Setup
To run the experiments, we need to install the required packages. A `env.yml` file is included in the repository to create a conda environment with the required packages. To create the environment, run the following command in the root directory:
```bash
conda env create -f env.yml
```

Then activate the environment with the following command:
```bash
conda activate drboot
```

## Running the Experiments
All experiments included in the thesis are stored in the `experiment_scripts/` directory.

For example, to run the APPS bootstrapping experiment with full compiler feedback, we can navigate to `experiment_scripts/apps_jobs/full_feedback_apps_job.sh` and run the following command in the root directory:
```bash
python train_sdr.py --batch-size-per-replica 6 --grad-acc-steps 4 --inference_batch_size 70 --num_workers 16 --model codet5-large-ntp-py --training_mode full_feedback --exp_name full_feedback_bootstrap_apps_1 --perform_experiments --beam_search_batch_size 35 --dataset APPS --only_perform_basic_tests --seed 18 --validate_first_step  --model codet5-large-ntp-py
```

## License
[Copyright (c) 2023 Qualcomm Innovation Center, Inc.](LICENSE)