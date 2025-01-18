import os
import tarfile
from pathlib import Path

from tqdm import tqdm

# def add_files(tar, path):
#     for item in path.iterdir():
#         if item.is_symlink():
#             tar.add(item.readlink())
#         elif item.is_dir():
#             add_files(tar, item)


def create_tar_with_symlinks(tar_name, source_dir):
    source_path = Path(source_dir)
    with tarfile.open(tar_name, "w") as tar:
        add_files(tar, source_path)


# Example usage
# create_tar_with_symlinks("example.tar", "models")


convnext_finetuning = {
    "convnext_finetuning_s:2_reg:False": "results/2024/05.14/13-28-06/7/cifar100_fixed_finetuning",
    "convnext_finetuning_s:1_reg:False": "results/2024/05.14/13-28-06/6/cifar100_fixed_finetuning",
    "convnext_finetuning_s:2_reg:True": "results/2024/05.14/13-27-53/13/cifar100_fixed_finetuning",
    "convnext_finetuning_s:1_reg:True": "results/2024/05.14/13-27-53/12/cifar100_fixed_finetuning",
    "convnext_finetuning_s:0_reg:True": "results/2024/04.24/13-34-44/0/cifar100_fixed_finetuning",
    "convnext_finetuning_s:0_reg:False": "results/2024/04.17/18-12-39/0/cifar100_fixed_finetuning_final_checkpoint",
}

finetuning = {
    "finetuning_first_task_reg": "results/2024/04.24/13-35-10/1",
    "resnet_finetuning_s:0_reg:True": "results/2024/04.24/13-35-10/0",
    "resnet_finetuning_s:0_reg:False": "results/2024/04.17/18-13-25/0",
    "resnet_finetuning_s:2_reg:False": "results/2024/05.14/13-28-06/1",
    "resnet_finetuning_s:1_reg:False": "results/2024/05.14/13-28-06/0",
    "resnet_finetuning_s:2_reg:True": "results/2024/05.14/13-27-53/19",
    "resnet_finetuning_s:1_reg:True": "results/2024/05.14/13-27-53/18",
}

# first_task = {
#     "finetuning_first_task_reg_s:1": "results/2024/05.21/18-49-03/0/cifar100_fixed_finetuning",
#     "finetuning_first_task_reg_s:2": "results/2024/05.21/18-49-03/1/cifar100_fixed_finetuning",
# }

first_task = {
    "finetuning_first_task_reg_s:1": "results/2024/05.22/09-47-57/0/cifar100_fixed_finetuning",
    "finetuning_first_task_reg_s:2": "results/2024/05.22/09-47-57/1/cifar100_fixed_finetuning",
}

# Output tar file name
tar_filename = "seed_first_task_new.tar.gz"


def add_files(tar, path: Path):
    for item in path.iterdir():
        if item.is_file():
            tar.add(item)
        elif item.is_dir():
            add_files(tar, item)


def create_tar(tar_name, source_dir):
    source_path = Path(source_dir)
    with tarfile.open(tar_name, "a") as tar:
        add_files(tar, source_path)


for name, path in tqdm(first_task.items()):
    create_tar(tar_filename, path)
