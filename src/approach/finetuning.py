from argparse import ArgumentParser
from typing import Optional

import omegaconf
import torch

from src.datasets.exemplars_dataset import ExemplarsDataset
from src.loggers.exp_logger import ExperimentLogger
from src.regularizers import VarCovRegLossInterface

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(
        self,
        model,
        device,
        varcov_regularizer: VarCovRegLossInterface,
        logger: Optional[ExperimentLogger] = None,
        exemplars_dataset: Optional[ExemplarsDataset] = None,
        *,
        cfg,
    ):
        super(Appr, self).__init__(
            model,
            device,
            varcov_regularizer,
            logger,
            exemplars_dataset,
            cfg=cfg,
        )

        self.set_methods_defaults(cfg.approach)

        self.all_out = cfg.approach.kwargs.all_outputs

    def set_methods_defaults(self, cfg: omegaconf.DictConfig):
        defaults = {
            "all_outputs": False,
        }
        if not cfg.kwargs:
            cfg.kwargs = omegaconf.DictConfig(defaults)
        else:
            for key in defaults.keys():
                cfg.kwargs.setdefault(key, defaults[key])

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self):
        """Returns the optimizer"""
        if (
            len(self.exemplars_dataset) == 0
            and len(self.model.heads) > 1
            and not self.all_out
        ):
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(
                self.model.heads[-1].parameters()
            )
        else:
            params = self.model.parameters()
        return self.optimizer_factory(params=params)

    # torch.optim.SGD(
    #         params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum
    #     )

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(
            outputs[t], targets - self.model.task_offset[t]
        )
