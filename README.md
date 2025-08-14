# Combining Machine Learning Defenses without Conflicts

Code for the paper titled "Combining Machine Learning Defenses without Conflicts" published in TMLR 2025.

## Requirements

You need __conda__. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate defense-interactions
```

To update the env:

```bash
conda env update --name defense-interactions --file environment.yml
```

or

```bash
conda activate defense-interactions
conda env update --file environment.yml
```

## Datasets

Link to UTKFACE dataset: https://drive.google.com/file/d/1iLCJEu2bwVdd0SiZFzNMkzvhH3TjTWN4/view?usp=sharing


## Evaluating Standalone Defenses

The script standalone_script.sh which contains all the commands to evaluate all defenses.

```bash
./standalone_script.sh 
```
Or you can run the commands from the script individually.

## Evaluating Pairwise Defense Combinations

The script pairwise_script.sh which contains all the commands to evaluate all defenses.

```bash
./pairwise_script.sh 
```
Or you can run the commands from the script individually.


## Hyperparameter tuning

The script hyperparams.sh which contains all the commands to evaluate all defenses.

```bash
./hyperparams.sh 
```

## Evaluating Multiple Defense Combinations

The script multiple_script.sh which contains all the commands to evaluate all defenses.

```bash
./multiple_script.sh 
```
Or you can run the commands from the script individually.


## Evaluating Evasion Attack Variants

For three combinations, C9, C12, and C35, we evaluate against PGD and AutoAttack using SecML library and also cover attack indicators.

We need a slightly different setup to run this code. 

As suggested in SecML documentation: https://secml.readthedocs.io/en/v0.15/#tutorials, we eed python 3.8 or less.

We tested on  3.7.16.

```bash
conda env create -f environment_evasion.yml
```

To run all experiments
```bash
./evasion.sh 
```
Or you can run the commands from the script individually.

## Credits

- Poisoning + Watermarking: https://github.com/verazuo/badnets-pytorch/tree/master
- Group fairness: https://github.com/ahxt/fair_fairness_benchmark
- Differential privacy: Opacus library
- Dataset Watermarking: https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/tree/main
- Adversarial training: Cleverhans library
- Model explanations: Captum library
- Outlier Removal: https://github.com/RJ-T/NIPS2022_EP_BNP
- Model watermarking (In processing): https://github.com/arpitbansal297/Certified_Watermarks
- Fingerprinting (Dataset Inference): https://github.com/ssg-research/amulet
