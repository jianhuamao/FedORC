# FedOCR

FedOCR is a one-shot client-block outlier removal method for robust federated learning.
Instead of correcting noisy samples during training, FedOCR filters risky client blocks before federated optimization starts.
It uses probing-set predictive entropy as a reliability signal and proxy distribution matching as a coverage signal, then keeps a retained client pool that is cleaner and more balanced.
The retained pool can be trained with FedAvg directly, or used as a front-end admission module before other federated learning methods.

## Why FedOCR

Federated learning often suffers from two practical issues at the same time: noisy supervision and heterogeneous client data.
FedOCR is designed for that setting.
Its goal is to remove the most harmful client blocks early, so the system spends less computation and communication on clients that would otherwise be corrected later anyway.
This makes FedOCR a practical pre-training filter rather than a sample-level correction method.

## Main idea

FedOCR follows a simple pipeline:

1. Each candidate client performs a short local adaptation step.
2. The server evaluates the adapted client model on a clean probing set.
3. Predictive entropy measures client reliability.
4. A proxy distribution score measures whether the retained pool still covers the target distribution well.
5. A greedy admission rule constructs the final client set.

In the paper, the standalone FedOCR rows correspond to `FedOCR + FedAvg`.
The `+FedAvg` part is omitted in tables and figures for brevity.

## Repository layout

```text
FedOCR/
  main.py                 # Main entry point
  train.py                # Federated training loop
  MS_environment.yml      # Conda environment specification
  lib/
    dataset/              # Data preparation, partitioning, and noise injection
    model/                # FedOCR selector and baseline methods
    utils.py              # Client construction and selection helpers
  train_exp/              # Example launch scripts used in experiments
  oracle/                 # Oracle experiments and theoretical validation logs
  result/                 # Generated logs, figures, and experiment outputs
```

## Environment

The project was developed with Python 3.12.
The easiest way to install dependencies is through the provided Conda file:

```bash
conda env create -f MS_environment.yml
conda activate MS
```

If you use a different Python installation, make sure the environment can import PyTorch, torchvision, NumPy, pandas, matplotlib, SciPy, and the other packages listed in the environment file.

## Quick start

The codebase keeps the historical command-line flag `--mode ours` for backward compatibility.
That flag now refers to the FedOCR admission module.

### Run FedOCR directly

```bash
python main.py \
  --mode ours \
  --data_name cifar10 \
  --noise_mode linear_noise \
  --device cuda:0
```

### Reproduce the paper-style launch scripts

The `train_exp/` folder contains the bash launchers used for the main experiments.
For example:

```bash
bash train_exp/train.sh --data_name cifar10 --noise_mode linear_noise --device cuda:0
bash train_exp/run_fedcorr_official_cifar10_linear_noise_gpu0.sh
bash train_exp/run_cifar10_fixed_scale_gpu2_parallel.sh
```

These scripts are designed for Linux or WSL environments.
Some of them use `screen` for long-running jobs.

## Supported experiment settings

The code supports the settings used in the paper:

- MedMNIST and CIFAR-10/100
- client-level symmetric label noise
- linear and Gaussian heterogeneous noise
- Dirichlet Non-IID partitioning
- FedAvg as the default downstream optimizer after FedOCR admission
- modular combinations with downstream noisy-label FL methods

## What each main module does

- `lib/model/fedocr.py`: the FedOCR selector and its scoring logic
- `lib/model/selection.py`: routing for FedOCR and baseline selection methods
- `lib/dataset/pre_data.py` and `lib/dataset/dataset.py`: data preparation, partitioning, and noise handling
- `lib/utils.py`: client creation, selection helpers, and logging utilities
- `train.py`: the main federated optimization loop
- `main.py`: experiment entry point and argument wiring

## Outputs

Generated logs, figures, and comparison artifacts are written to `result/`.
The `oracle/` directory stores the controlled oracle experiments and the theoretical validation plots used in the paper.

## Notes

- Some legacy log folders may still contain older names from earlier development stages.
  The current implementation uses the FedOCR name in the code path.
- The standalone FedOCR results in the paper are `FedOCR + FedAvg`.
- If you want a clean rerun, start from an empty `result/` folder so old outputs do not mix with new ones.

## Citation

If you use this code, please cite the FedOCR paper.

