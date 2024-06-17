# Repository

This repository is based on the codebase of [FACIL](https://github.com/mmasana/FACIL) and [Adapt Your Teacher](https://github.com/fszatkowski/cl-teacher-adaptation). Please refer to the `/experiments` directory for the training scripts used in the paper.

To reproduce FeTRIL, add [VarCovRegLoss](src/regularizers/__init__.py) to [PYCIL](https://github.com/G-U-N/PyCIL) similarly.

## Reproducing Main Results

```
python -m src.main_incremental -cp ../experiments/[cifar|imagenet] -cn [proper_yaml_file]
```

Example:

```
python -m src.main_incremental -cp ../experiments/cifar -cn reg_imagenet_10_20_tasks
```

## Minimal Working Example

```
python -m src.main_incremental misc.results_path=results
```

# Requirements

- python3.10
- hydra-core
- hydra-submitit-launcher 
- torch
- torchvision
- matplotlib
- numpy
- seaborn
- tensorboard
- wandb
- python-dotenv
- imagecorruptions

# Dataset

CIFAR will be downloaded automatically if needed. For ImageNet100, download [Kaggle ImageNet100](https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn) and extract it to `/data`.


# License

MIT
