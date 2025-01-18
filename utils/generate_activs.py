import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.analysis import core

get_digits = lambda str: int(re.sub(r"\D", "", str))


def get_activs(exp_name, path):
    root = Path("activations")
    save_path = root.joinpath(*exp_name.split("_"))
    save_path.mkdir(exist_ok=True, parents=True)

    train_tasks_data = defaultdict(dict)
    test_tasks_data = defaultdict(dict)
    cfg = core.create_cfg(path)
    cfg.data.num_workers = 16
    data_factory = core.DataFactory(cfg)
    model_factory = core.ModelFactory(cfg, path, device=DEVICE)
    for task_idx, ckpt in tqdm(enumerate(sorted(model_factory.ckpts))):
        model = model_factory.create_model(task=task_idx, num_classes=NUM_CLASSES)
        assert task_idx == get_digits(ckpt.name)
        train_loaders, _, test_loaders, _ = data_factory[: task_idx + 1]
        for data_idx, (train_loader, test_loader) in enumerate(
            zip(train_loaders, test_loaders)
        ):
            train_outs = core.get_activations(model.model, train_loader, device=DEVICE)
            test_outs = core.get_activations(model.model, test_loader, device=DEVICE)

            train_tasks_data[task_idx][data_idx] = train_outs
            test_tasks_data[task_idx][data_idx] = test_outs
    torch.save(train_tasks_data, save_path / "train.pth")
    torch.save(test_tasks_data, save_path / "test.pth")
    return train_tasks_data, test_tasks_data


if __name__ == "__main__":
    DEVICE = "cuda"
    NUM_CLASSES = 20

    resnet_finetuning = {
        "resnet34_finetuning_reg": "results/2024/04.24/13-35-10/0/cifar100_fixed_finetuning",
        "resnet34_finetuning_noreg": "results/2024/04.17/18-13-25/0/cifar100_fixed_finetuning_final_checkpoint",
    }

    convnext_finetuning = {
        "convnext_finetuning_s:2_reg:False": "results/2024/05.14/13-28-06/7/cifar100_fixed_finetuning",
        "convnext_finetuning_s:1_reg:False": "results/2024/05.14/13-28-06/6/cifar100_fixed_finetuning",
        "convnext_finetuning_s:2_reg:True": "results/2024/05.14/13-27-53/13/cifar100_fixed_finetuning",
        "convnext_finetuning_s:1_reg:True": "results/2024/05.14/13-27-53/12/cifar100_fixed_finetuning",
        "convnext_finetuning_s:0_reg:True": "results/2024/04.24/13-34-44/0/cifar100_fixed_finetuning",
        "convnext_finetuning_s:0_reg:False": "results/2024/04.17/18-12-39/0/cifar100_fixed_finetuning_final_checkpoint",
    }

    for e_name, path in convnext_finetuning.items():
        tasks_data = get_activs(exp_name=e_name, path=path)
