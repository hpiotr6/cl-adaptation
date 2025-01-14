import pandas as pd
import wandb

from itertools import product
from matplotlib import pyplot as plt
import seaborn as sns

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("tunnels-ssl/07.04")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    if run.state == "running":
        continue
    summary_list.append(run.summary._json_dict["test/avg_acc_tag"])

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"avg_acc_tag": summary_list, "config": config_list, "name": name_list}
)

config_df = pd.json_normalize(runs_df["config"])
df = pd.concat([runs_df.drop(columns=["config"]), config_df], axis=1)
df["data.datasets"] = df["data.datasets"].apply(lambda x: x[0])
from typing import Optional


layers = {
    "last_layer": ["fc", "classifier"],
    "intermediate": ["after_relu", "after_skipping"],
}


def get_df(
    df,
    network: str,
    layer: str,
    dataset: str,
    is_scaled: bool,
    noreg_val: Optional[float] = None,
):
    conditions = (
        (df["model.network"].str.contains(network))
        & (df["misc.seed"] == 0)
        & (df["data.datasets"].str.contains(dataset))
        & (df["training.approach.name"] == "finetuning")
        & (df["data.exemplars.num_exemplars"] == 0)
        & (df["data.stop_at_task"] == 3)
        & (df["data.num_tasks"] == 10)
    )
    varcov = conditions & (
        df["training.vcreg.reg_layers"].str.contains("|".join(layer))
        & (df["training.vcreg.scale"] == is_scaled)
    )

    if noreg_val is None:
        noreg = conditions & (df["training.vcreg.reg_layers"].isna())
        noreg_val = df[noreg]["avg_acc_tag"].item()

    filtered_df = df[varcov]
    assert filtered_df.shape[0] == 16
    filtered_df["diff"] = filtered_df["avg_acc_tag"] - noreg_val
    return filtered_df


networks = ["resnet", "convnext"]
datasets = ["imagenet", "cifar"]
layers = layers["intermediate"]
is_scaled = [True, False]
n_rows = len(networks)
n_cols = len(is_scaled) * len(datasets)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

axes = axes.flatten()

for ax, (network, layer, scaled, dataset) in zip(
    axes, product(networks, layers, is_scaled, datasets)
):
    # Get the DataFrame for the specific network and layer

    p = get_df(df, network, layer, dataset, scaled, 0).pivot(
        index="training.vcreg.var_weight",
        columns="training.vcreg.cov_weight",
        values="diff",
    )
    # Plot the heatmap on the current axis
    sns.heatmap(p, annot=True, ax=ax, vmin=-10, vmax=10)
    ax.set_title(" ".join([network, layer]))

# Adjust layout
plt.tight_layout()
plt.show()
