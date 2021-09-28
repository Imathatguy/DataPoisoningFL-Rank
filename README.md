# MANDERA: Malicious Node Detection in Federated Learning via Ranking

Code for a ICLR 2021 submission, "MANDERA: Malicious Node Detection in Federated Learning via Ranking"

The code in this repository has been adapted from code originally from the ESORICS 2020 paper: Data Poisoning Attacks Against Federated Learning Systems

## Installation

1) Create a virtualenv (Python 3.7)
2) Install dependencies inside of virtualenv (```pip install -r requirements.pip```)
3) If you are planning on using the defense, you will need to install ```matplotlib```. This is not required for running experiments, and is not included in the requirements file

## Instructions for execution

Using this repository, you can replicate all results presented at ESORICS. We outline the steps required to execute different experiments below.

### Setup

Before you can run any experiments, you must complete some setup:

1) ```python3 generate_data_distribution.py``` This downloads the datasets, as well as generates a static distribution of the training and test data to provide consistency in experiments.
2) ```python3 generate_default_models.py``` This generates an instance of all of the models used in the paper, and saves them to disk.

### General Information

Some pointers & general information:
- Most hyperparameters can be set in the ```federated_learning/arguments.py``` file
- Most specific experiment settings are located in the respective experiment files (see the following sections)

### Experiments - Malicious Node detection by MANDERA

Running an attack: 
```
cd process_results
python3 process_results.py
```
or
```
python3 process_results_unzipped.py
```
Depending on the state of the gradients saved from the experiments below:
Note that gradient saving requires a substantial amount of disk space as has been supressed by default.


### Experiments - MANDERA for defending against poisoning attacks
To run the full set of experiments for each of the 4 attacks and defenses
See the respective bash script `label_flipping_batch.sh`, `guassian_attack.sh`, `zero_gradient_attack.sh`, `sign_flipping_batch.sh`

The batch experiments were run on a HPC enironment with a slurm scheduller.

Running an attack: ```bash label_flipping.batch.sh```


### Experiments - Computational Efficiency

Running a timing test:
```
cd timing_test
slurm timing.jobscript
```

### Experiment Hyperparameters

Recommended default hyperparameters:
- Batch size: 10
- LR: 0.01
- Number of epochs: 200
- Momentum: 0.5
- Scheduler step size: 50
- Scheduler gamma: 0.5
- Min_lr: 1e-10
