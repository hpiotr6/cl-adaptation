from collections import defaultdict
from datetime import date
from functools import partial
import os
from pathlib import Path
from pprint import pprint
from typing import Any
import torch

from src.metrics import cm
from src.sign_visualizer.core import (
    get_activations,
    load_and_clean,
    load_extractor,
    load_whole_model,
)
from src.sign_visualizer.dataset_interface import (
    ContinualDataset,
    ContinualDatasetConfig,
)


def is_sign_changed(output_1: torch.Tensor, output_2: torch.Tensor):
    return (output_1.mean(0) * output_2.mean(0)) < 0


def is_negative(output_1: torch.Tensor):
    return output_1.mean(0) < 0


def save(container: Any, path: str) -> None:
    torch.save(container, path)


if __name__ == "__main__":

    # path = "results/2024/03.07/cifar100_fixed_finetuning_after_relu"
    # path = "results/2024/03.07/cifar100_fixed_finetuning_fc"
    # path = "results/2024/03.07/cifar100_fixed_finetuning_before_relu"
    paths = [
        # "results/2024/03.21/cifar10_fixed_finetuning",
        # "results/2024/03.22/cifar10_fixed_lwf",
        "results/2024/03.21/cifar100_fixed_ewc",
        "results/2024/03.21/cifar100_fixed_finetuning",
        "results/2024/03.21/cifar100_fixed_lwf",
        "results/2024/03.22/cifar100_fixed_finetuning",
    ]
    for path in paths:
        NUM_TASKS = 5
        NUM_CLASSES = 20
        DEVICE = "cpu"
        # path = (
        #     "results/2024/03.14/cifar10_fixed_finetuning_.*after_relu|fc$:var_0.64:cov_12.8"
        # )
        model = partial(load_whole_model, path=path)
        config = ContinualDatasetConfig("cifar100_fixed", NUM_TASKS)
        cl_dataset = ContinualDataset(config)
        dataloader = partial(
            torch.utils.data.DataLoader,
            batch_size=64,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )

        tst_loaders = [dataloader(cl_dataset[i, "test"]) for i in range(NUM_TASKS)]

        results = []
        for task_id in range(NUM_TASKS):
            current_model = model(task=task_id, num_classes=NUM_CLASSES)
            result = cm(
                current_model,
                tst_loaders[: task_id + 1],
                NUM_TASKS,
                DEVICE,
            )
            results.append(result)
            pprint(result)

        save_path = Path("cms", *Path(path).parts[1:])
        os.makedirs(save_path, exist_ok=True)
        save(
            results,
            os.path.join(save_path, "cm.pkl"),
        )
