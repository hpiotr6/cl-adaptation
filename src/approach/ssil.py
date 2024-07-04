import warnings
import torch
from copy import deepcopy
from argparse import ArgumentParser
import torch.nn.functional as F
from typing import Optional
import omegaconf

from src.loggers.exp_logger import ExperimentLogger

from .incremental_learning import Inc_Learning_Appr
from src.datasets.exemplars_dataset import ExemplarsDataset
from src.regularizers import VarCovRegLossInterface


class Appr(Inc_Learning_Appr):
    """Class implementing the SS-IL : Separated Softmax for Incremental Learning approach
    described in:
    https://openaccess.thecvf.com/content/ICCV2021/papers/Ahn_SS-IL_Separated_Softmax_for_Incremental_Learning_ICCV_2021_paper.pdf

    Code: https://github.com/hongjoon0805/SS-IL-Official/blob/master/trainer/ssil.py
    """

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

        self.model_old = None
        self.lamb = cfg.approach.kwargs.lamb
        self.T = cfg.approach.kwargs.T
        self.replay_batch_size = cfg.approach.kwargs.replay_batch_size

        self.ta = cfg.approach.kwargs.ta

        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")

        # SSIL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        if not have_exemplars:
            warnings.warn(
                "Warning: SS-IL is expected to use exemplars. Check documentation."
            )

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def set_methods_defaults(self, cfg: omegaconf.DictConfig):
        defaults = {
            "lamb": 1,
            "T": 2,
            "ta": False,
            "replay_batch_size": 32,
        }
        if not cfg.kwargs:
            cfg.kwargs = omegaconf.DictConfig(defaults)
        else:
            for key in defaults.keys():
                cfg.kwargs.setdefault(key, defaults[key])

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        exemplar_selection_loader = torch.utils.data.DataLoader(
            trn_loader.dataset + self.exemplars_dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        self.exemplars_dataset.collect_exemplars(
            self.model, exemplar_selection_loader, val_loader.dataset.transform
        )

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        if t > 0:
            exemplar_loader = torch.utils.data.DataLoader(
                self.exemplars_dataset,
                batch_size=self.replay_batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True,
            )
            trn_loader = zip(trn_loader, exemplar_loader)

        self.model.train()
        if self.ta and self.model_old is not None:
            self.model_old.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for samples in trn_loader:
            if t > 0:
                (data, target), (data_r, target_r) = samples
                data, data_r = data.to(self.device), data_r.to(self.device)
                data = torch.cat((data, data_r))
                target, target_r = target.to(self.device), target_r.to(self.device)
                # Forward old model
                targets_old = self.model_old(data.to(self.device))
            else:
                data, target = samples
                data = data.to(self.device)
                target = target.to(self.device)
                target_r = None
                targets_old = None
            # Forward current model
            var_loss, cov_loss, feats = self.varcov_regularizer(
                self.model.model, data, t
            )
            outputs = [head(feats) for head in self.model.heads]
            varcov_loss = (
                var_loss * self.varcov_regularizer.vcr_var_weight
                + cov_loss * self.varcov_regularizer.vcr_cov_weight
            )

            loss = self.criterion(t, outputs, target, target_r, targets_old)
            loss += varcov_loss.mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def criterion(self, t, outputs, target, target_r=None, outputs_old=None):
        loss_KD = 0
        batch_size = len(target)
        replay_size = len(target_r) if target_r is not None else 0
        loss_CE_curr = self.loss(
            outputs[t][:batch_size], target - self.model.task_offset[t]
        )

        if t > 0 and target_r is not None:
            prev = torch.cat(
                [o[batch_size : batch_size + replay_size] for o in outputs[:t]], dim=1
            )
            loss_CE_prev = self.loss(prev, target_r)
            loss_CE = (loss_CE_curr + loss_CE_prev) / (batch_size + replay_size)

            # loss_KD
            loss_KD = torch.zeros(t).to(self.device)
            for _t in range(t):
                soft_target = F.softmax(outputs_old[_t] / self.T, dim=1)
                output_log = F.log_softmax(outputs[_t] / self.T, dim=1)
                loss_KD[_t] = F.kl_div(
                    output_log, soft_target, reduction="batchmean"
                ) * (self.T**2)
            loss_KD = loss_KD.sum()
        else:
            loss_CE = loss_CE_curr / batch_size

        return loss_CE + self.lamb * loss_KD
