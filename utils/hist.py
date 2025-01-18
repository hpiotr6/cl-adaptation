from pathlib import Path

import torch
from matplotlib import pyplot as plt

ROOT = "analysis/12.09"
directory_path = Path(ROOT)
task_1_paths = list(directory_path.rglob("task_1*"))
task_2_paths = list(directory_path.rglob("task_2*"))
assert len(task_1_paths) == 2
assert len(task_2_paths) == 2


def get_W_b(path):
    ckpt = torch.load(path, map_location="cpu")["state_dict"]
    W = ckpt["linear_layer.weight"].flatten()
    b = ckpt["linear_layer.bias"]
    return W, b


def diff(paths):
    W1, b1 = get_W_b(paths[0])
    W2, b2 = get_W_b(paths[1])
    return W2 - W1, b2 - b1


for idx, (task1, task2) in enumerate(zip(task_1_paths, task_1_paths)):
    wd, bd = diff((task1, task2))
    plt.hist(wd, bins=100, alpha=0.5)
    plt.savefig(f"diff{idx}.png")
    break
