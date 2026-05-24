# Neural Network and Deep Learning Project 1

This repository contains the NumPy implementation for Project 1: MNIST handwritten digit classification with an MLP baseline and a simple CNN.

## Contents

- `codes/mynn/`: NumPy neural-network operators, models, optimizer, scheduler, metric, and runner.
- `codes/run_experiments.py`: trains the MLP baseline, CNN with SGD, and CNN with Momentum + learning-rate scheduling.
- `codes/sanity_checks.py`: gradient checks for `Linear` and `conv2D`.
- `codes/test_train.py`: training entry point.
- `codes/test_model.py`: saved-model evaluation entry point.
- `codes/make_report.py`: regenerates the local Word report from saved experiment results.

## Notes

The MNIST dataset, trained checkpoints, generated figures, and other large files are intentionally not included in this repository, following the course submission requirement.
