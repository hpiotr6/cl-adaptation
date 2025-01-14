import pandas as pd
import wandb


class WandbWrapper:
    def __init__(self) -> None:
        self.api = wandb.Api()
        self.crashed = []

    def create_df_from_project(self, project_name: str) -> pd.DataFrame:
        runs = self.api.runs(project_name)

        summary_list, config_list, name_list = [], [], []
        for run in runs:
            if run.state == "running":
                continue

            elif (
                run.state == "crashed"
                or len(run.history(keys=["test/avg_acc_tag"]))
                != run.config["data"]["num_tasks"]
            ):
                self.crashed.append(run.name)
                continue

            summary_list.append(run.summary._json_dict["test/avg_acc_tag"])
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            name_list.append(run.name)

        runs_df = pd.DataFrame(
            {"avg_acc_tag": summary_list, "config": config_list, "name": name_list}
        )

        config_df = pd.json_normalize(runs_df["config"])
        df = pd.concat([runs_df.drop(columns=["config"]), config_df], axis=1)
        df["data.datasets"] = df["data.datasets"].apply(lambda x: x[0])
        self.add_real_name(df)

        return df

    @staticmethod
    def add_real_name(df) -> None:
        def get_real_name(row):
            if row["data.exemplars.num_exemplars"] > 0:
                return "replay"
            return row["training.approach.name"]

        df["real_name"] = df.apply(get_real_name, axis=1)


# wrapper = WandbWrapper()
# old_df = wrapper.create_df_from_project("tunnels-ssl/07.18")
