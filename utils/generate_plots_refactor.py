import os
from pathlib import Path

import hydra
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig


class Plot:
    def __init__(self, title, save_dir) -> None:
        sns.set_theme()
        self._x = None
        self._upper_bound = None
        self.save_dir = save_dir
        self.title = title
        self.fig, self.ax1 = plt.subplots(1, 1, figsize=(15, 10))

        self.ax2 = self.ax1.twinx()
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel("Accuracy")
        self.ax2.grid(False)

        self.ax1.set_ylim(0, 8200)
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Rank")
        self.ax1.tick_params(axis="y")
        self.ax1.tick_params(axis="x", rotation=90)
        self.ax1.grid(True)

    @property
    def x(self):
        return [name.replace("backbone.", "") for name in self._x]

    @x.setter
    def x(self, val):
        if self._x is None:
            self._x = val
        else:
            assert self._x == val

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, val):
        if self._upper_bound is None:
            self._upper_bound = val
            self.ax1.plot(self.x, self._upper_bound, "k--", label="Rank upper bound")
        else:
            assert self._upper_bound == val

    def add_spectra(self, spectra, upperbound=None, fmt=None, *args, **kwargs):
        if upperbound is not None:
            self.upper_bound = upperbound
        self.ax1.plot(self.x, spectra, fmt, *args, **kwargs)

    def add_acc(self, early, fmt=None, *args, **kwargs):
        self.ax2.plot(self.x, early, fmt, *args, **kwargs)

    def save_fig(self):
        lines, labels = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc=0)

        self.fig.suptitle(self.title)
        # plt.subplots_adjust(bottom=0.1)
        self.fig.tight_layout()
        self.fig.savefig(os.path.join(self.save_dir + f"{self.title}.png"), dpi=300)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot_it(cfg: DictConfig):
    root = Path(cfg.root)
    title = f"{root.name}".replace("_\d", "")
    save_dir = f"{os.path.join(*root.parts[:2])}/plots/"
    os.makedirs(save_dir, exist_ok=True)
    plot = Plot(title=title, save_dir=save_dir)

    for sub_plot in cfg.plots:
        path = os.path.join(cfg.root, sub_plot.filename)
        checkpoint = torch.load(path)
        match sub_plot.type:
            case "representation_spectra":
                # spectra = list(checkpoint["rank"].values())
                # bound = list(checkpoint["upper_bound"].values())

                plot.x = list(checkpoint["rank"].keys())
                plot.x = list(checkpoint["upper_bound"].keys())
                spectra = values_from(checkpoint["rank"])
                bound = values_from(checkpoint["upper_bound"])
                plot.add_spectra(spectra, bound, **sub_plot.plot_kwargs)
            case "early_exit":
                # exit = list(checkpoint.values())
                plot.x = list(checkpoint.keys())
                exit = values_from(checkpoint)
                plot.add_acc(exit, **sub_plot.plot_kwargs)
            case _:
                raise ValueError(f"Unknown plot type: {sub_plot.type}")

    plot.save_fig()


def values_from(tensor_dict):
    return [val for val in tensor_dict.values()]


if __name__ == "__main__":
    plot_it()
