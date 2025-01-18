import os
from collections import defaultdict
from datetime import date
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any

import torch

from src.sign_visualizer.core import (choose_task, get_activations,
                                      load_and_clean, load_extractor)
from src.sign_visualizer.dataset_interface import (ContinualDataset,
                                                   ContinualDatasetConfig)


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
        # path = (
        #     "results/2024/03.14/cifar10_fixed_finetuning_.*after_relu|fc$:var_0.64:cov_12.8"
        # )
        model = partial(load_extractor, path=path, load_and_clean=load_and_clean)
        config = ContinualDatasetConfig("cifar100_fixed", 5)
        cl_dataset = ContinualDataset(config)
        dataloader = partial(
            torch.utils.data.DataLoader,
            batch_size=64,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )

        results = {}
        for task_id in range(0, 4):
            test_0 = cl_dataset[task_id, "test"]
            test_0_dataloader = dataloader(test_0)
            activations_0 = get_activations(model(task=task_id), test_0_dataloader)

            changes = defaultdict(list)

            for i in range(task_id + 1, 5):
                activations_1 = get_activations(model(task=i), test_0_dataloader)
                for label in activations_0.keys():

                    d_changed = is_sign_changed(
                        activations_0[label], activations_1[label]
                    )

                    # d_changed = is_negative(activations_0[label])
                    changes[label].append(d_changed.sum().item() / 512)
                    print(f"{label} direction changed in {d_changed.sum().item()}")
                print(f"Task {i} {'#'*20}")

            results[task_id] = changes

        save_path = Path("sign_mass_results", *Path(path).parts[1:])
        os.makedirs(save_path, exist_ok=True)

        save(
            results,
            os.path.join(save_path, "sign.pkl"),
        )
        pprint(results)
