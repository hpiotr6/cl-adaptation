from datetime import datetime
import pandas as pd
import wandb


def get_df_cl(project_name: str, min_runtime_minutes: int = 5):
    api = wandb.Api()
    entity = "tunnels-ssl"
    # Get all runs for the specified project
    runs = api.runs(f"{entity}/{project_name}")

    min_runtime_seconds = 60 * min_runtime_minutes
    failed_runs = []
    not_finished_runs = []
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        start_time_str = run.createdAt
        end_time_str = (
            run.heartbeatAt if run.heartbeatAt else run.updatedAt
        )  # Fallback to updatedAt if heartbeatAt is not available
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))

        # Calculate the difference
        runtime_seconds = (end_time - start_time).total_seconds()

        if runtime_seconds <= min_runtime_seconds:
            failed_runs.append(run.name)
            continue

        if run.state != "finished":
            not_finished_runs.append(run.name)
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict["test/avg_acc_tag"])

        # __import__("pdb").set_trace()
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"avg_acc_tag": summary_list, "config": config_list, "name": name_list}
    )
    config_df = pd.json_normalize(runs_df["config"])
    df = pd.concat([runs_df.drop(columns=["config"]), config_df], axis=1)
    df["data.dataset"] = df["data.datasets"].apply(lambda x: x[0])

    def get_real_name(row):
        if row["data.exemplars.num_exemplars"] > 0:
            return "replay"
        return row["training.approach.name"]

    def is_regularized(row):
        if (
            row["training.vcreg.var_weight"] == 0
            and row["training.vcreg.cov_weight"] == 0
        ):
            return False
        return True

    df["approach"] = df.apply(get_real_name, axis=1)
    df["is_regularized"] = df.apply(is_regularized, axis=1)
    df["is_big_task"] = df["data.nc_first_task"].map(lambda x: x == 50.0)
    df = df.rename(
        columns={
            # "test_acc": "avg_acc_tag",
            "data.num_tasks": "num_tasks",
            "data.dataset": "dataset",
            "model.network": "network",
            "misc.seed": "seed",
        }
    )

    df["dataset"] = df["dataset"].replace(
        {
            "cifar100_fixed": "cifar100",
            "imagenet_subset_kaggle": "imagenet100",
        }
    )

    df["project_name"] = project_name
    return df, failed_runs, not_finished_runs


def get_df_pycil(project_name: str, min_runtime_minutes: int = 5):
    api = wandb.Api()
    entity = "tunnels-ssl"
    runs = api.runs(f"{entity}/{project_name}")

    min_runtime_seconds = min_runtime_minutes * 60
    failed_runs = []
    not_finished_runs = []
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        start_time_str = run.createdAt
        end_time_str = (
            run.heartbeatAt if run.heartbeatAt else run.updatedAt
        )  # Fallback to updatedAt if heartbeatAt is not available
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))

        # Calculate the difference
        runtime_seconds = (end_time - start_time).total_seconds()

        if runtime_seconds <= min_runtime_seconds:
            failed_runs.append(run.name)
            continue
        if run.state != "finished":
            not_finished_runs.append(run.name)
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict["avg_acc_tag"])

        # __import__("pdb").set_trace()
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"avg_acc_tag": summary_list, "config": config_list, "name": name_list}
    )
    config_df = pd.json_normalize(runs_df["config"])
    df = pd.concat([runs_df.drop(columns=["config"]), config_df], axis=1)

    def is_regularized(row):
        if row["vcreg.var_weight"] == 0 and row["vcreg.cov_weight"] == 0:
            return False
        return True

    # df["approach"] = df.apply(get_real_name, axis=1)
    df["is_regularized"] = df.apply(is_regularized, axis=1)
    df["num_tasks"] = df["increment"].map(lambda x: int(100 / x))
    df["is_big_task"] = df["init_cls"].map(lambda x: x == 50.0)
    df = df.rename(
        columns={
            # "test_acc": "avg_acc_tag",
            "model_name": "approach",
            "convnet_type": "network",
        }
    )

    df["project_name"] = project_name
    return df, failed_runs, not_finished_runs
