from functools import partial
import os
from pathlib import Path
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()


def get_arrays(pattern, root):
    # dirs = [direct for direct in os.listdir(root) if re.match(pattern, direct)]
    # assert len(dirs) > 0, "Check pattern regex"

    FILENAMES = [
        "early_exit_from_0.pt",
        "early_exit_from_1.pt",
        "representation_spectra_from_0.pt",
        "representation_spectra_from_1.pt",
    ]
    repr = []
    repr_1 = []
    early_ood = []
    early = []
    for filename in FILENAMES:
        path = os.path.join(root, filename)
        results = torch.load(path)
        if "repr" in filename and "from_0" in filename:
            repr.append(list(results["rank"].values()))
            bound = list(results["upper_bound"].values())
        elif "repr" in filename and "from_1" in filename:
            repr_1.append(list(results["rank"].values()))
        elif "early" in filename and "from_1" in filename:
            early_ood.append(list(results.values()))
        else:
            early.append(list(results.values()))
            x = list(results.keys())

    repr = np.asarray(repr)
    repr_1 = np.asarray(repr_1)
    early_ood = np.asarray(early_ood)
    early = np.asarray(early)

    return x, repr, repr_1, early, early_ood, bound


def plot_muliplot(x, repr, repr_1, early, early_ood, bound, title, save_dir):
    lower_bound = lambda x: x.mean(0) - x.std(0)
    upper_bound = lambda x: x.mean(0) + x.std(0)
    x = [name.replace("backbone.", "") for name in x]
    fill_between = partial(plt.fill_between, x, alpha=0.3)
    # Create a figure
    # fig, ax1 = plt.subplots()
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    ax1.set_ylim(0, 8200)

    # Plot rank on the left y-axis
    ax1.plot(x, repr.mean(0), "bo-", label="Rank 0")
    ax1.plot(x, bound, "k--", label="Rank upper bound")
    fill_between(lower_bound(repr), upper_bound(repr), color="b")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Rank")
    ax1.tick_params(axis="y")

    ax1.plot(x, repr_1.mean(0), "ko--", label="Rank 1")

    # Create a second y-axis for accuracy on the right
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1)
    ax2.plot(x, early.mean(0), "ro-", label="Acc 0 ")
    fill_between(lower_bound(early), upper_bound(early), color="red")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis="y")

    ax2.plot(x, early_ood.mean(0), "go-", label="Acc 1")
    fill_between(lower_bound(early_ood), upper_bound(early_ood), color="g")

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    # Show the plot
    ax1.tick_params(axis="x", rotation=90)

    ax1.grid(True)
    ax2.grid(False)
    plt.title(title)
    # plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir + f"/{title}.png"), dpi=300)
    # plt.show()


if __name__ == "__main__":
    # dirs = ["barlow_twins", "simclr"]
    dirs = ["vicreg"]
    # patterns = [
    #     # "resnet34_\d",
    #     # "resnet34_nomodify_noskips",
    #     r"vicreg_resnet34_nomodify_skips",
    #     r"vicreg_resnet34_modify_noskips",
    #     r"vicreg_resnet34_modify_skips",
    #     r"vicreg_vgg19bn",
    # ]
    dirs = [
        "cifar100_finetuning_0_noreg",
        "cifar100_finetuning_1_fixed_noreg",
        "cifar100_finetuning_1_fixed_reg",
        "cifar100_finetuning_0_reg",
    ]
    for dir in dirs:
        # for pattern in patterns:
        root = os.path.join("analysis/11.22/2_tasks", dir)
        # root = "results/09.24/barlow_twins"
        title = f"{Path(root).name}".replace("_\d", "")
        x, repr, repr_1, early, early_ood, bound = get_arrays(dir, root)
        root_parts = Path(root).parts
        save_dir = f"{os.path.join(*root_parts[:2])}/plots/"
        os.makedirs(save_dir, exist_ok=True)
        plot_muliplot(x, repr, repr_1, early, early_ood, bound, title, save_dir)
