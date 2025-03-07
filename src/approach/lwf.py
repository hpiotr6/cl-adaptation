from argparse import ArgumentParser
from copy import deepcopy
from typing import Optional

import omegaconf
import torch

from src.datasets.exemplars_dataset import ExemplarsDataset
from src.loggers.exp_logger import ExperimentLogger
from src.metrics import cka
from src.regularizers import VarCovRegLossInterface

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
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
        self.mc = cfg.approach.kwargs.mc
        self.taskwise_kd = cfg.approach.kwargs.taskwise_kd

        self.ta = cfg.approach.kwargs.ta

        self.tp = cfg.approach.kwargs.tp
        self.ctt = cfg.approach.kwargs.ctt
        self.bnp = cfg.approach.kwargs.bnp
        self.cbnt = cfg.approach.kwargs.cbnt
        if self.tp or self.ctt or self.bnp or self.cbnt:
            assert not self.ta, "Cannot use both TA and TP/CTT/BNP/CBNT"
        if sum([self.tp, self.ctt, self.bnp, self.cbnt]) > 1:
            assert not self.ta, "Cannot use both TP and CTT and BNP and CBNT"
        self.pretraining_epochs = cfg.approach.kwargs.pretraining_epochs
        self.ta_lr = cfg.approach.kwargs.ta_lr

        self.cka = cka
        self.debug_loss = cfg.approach.kwargs.debug_loss

    def set_methods_defaults(self, cfg: omegaconf.DictConfig):
        defaults = {
            "lamb": 1,
            "T": 2,
            "mc": False,
            "taskwise_kd": False,
            "ta": False,
            "tp": False,
            "ctt": False,
            "bnp": False,
            "cbnt": False,
            "pretraining_epochs": 5,
            "ta_lr": 1e-5,
            "cka": False,
            "debug_loss": False,
        }
        if not cfg.kwargs:
            cfg.kwargs = omegaconf.DictConfig(defaults)
        else:
            for key in defaults.keys():
                cfg.kwargs.setdefault(key, defaults[key])

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def pre_train_process(self, t, trn_loader):
        if t > 0:
            if self.ctt or self.cbnt or self.tp or self.bnp:
                raise NotImplementedError
                self.model_old.add_head(self.model.heads[-1].out_features)
                self.model_old = self.model_old.to(self.device)

            if self.ctt or self.cbnt:
                raise NotImplementedError
                if self.ctt:
                    self.model_old.unfreeze_all()
                if self.cbnt:
                    self.model_old.unfreeze_bn()

                params = [p for p in self.model_old.parameters() if p.requires_grad]
                self.optimizer_old = torch.optim.SGD(
                    params, lr=self.ta_lr, weight_decay=self.wd
                )

            if self.tp or self.bnp:
                raise NotImplementedError
                if self.tp:
                    self.model_old.unfreeze_all()
                if self.bnp:
                    self.model_old.unfreeze_bn()
                self.model_old.train()

                params = [p for p in self.model_old.parameters() if p.requires_grad]
                self.optimizer_old = torch.optim.SGD(
                    params, lr=self.ta_lr, weight_decay=self.wd
                )

                for epoch in range(self.pretraining_epochs):
                    for images, targets in trn_loader:
                        images, targets = (
                            images.to(self.device),
                            targets.to(self.device),
                        )
                        outputs = self.model_old(images)
                        loss = self.criterion(t, outputs, targets)

                        self.optimizer_old.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model_old.parameters(), self.clipgrad
                        )
                        self.optimizer_old.step()

                self.model_old.freeze_all()
                self.model_old.eval()
        super().pre_train_process(t, trn_loader)

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

        self.training = True
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)
        self.training = False

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            # Forward old model
            targets_old = None
            if t > 0:
                if self.ctt or self.cbnt or self.ta:
                    self.model_old.train()
                else:
                    self.model_old.eval()

                targets_old = self.model_old(images)
                if self.ctt or self.cbnt:
                    raise NotImplementedError
                    loss_old = self.criterion(t, targets_old, targets)
                    self.optimizer_old.zero_grad()
                    loss_old.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model_old.parameters(), self.clipgrad
                    )
                    self.optimizer_old.step()
                    with torch.no_grad():
                        self.model_old.eval()
                        targets_old = self.model_old(images)

            loss, loss_kd, loss_ce = self.add_varcov_loss_lwf(
                self.model, t, images, targets, targets_old
            )

            if self.debug_loss:
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_kd",
                    group=f"debug_t{t}",
                    value=float(loss_kd),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_ce",
                    group=f"debug_t{t}",
                    value=float(loss_ce),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_total",
                    group=f"debug_t{t}",
                    value=float(loss),
                )

            if torch.isnan(loss):
                raise ValueError("Loss is NaN! Training cannot continue.")

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def add_varcov_loss_lwf(self, model, t, images, targets, targets_old):
        var_loss, cov_loss, corrs, feats = self.varcov_regularizer(
            model.model, images, t
        )
        outputs = [head(feats) for head in model.heads]
        varcov_loss = (
            var_loss * self.varcov_regularizer.vcr_var_weight
            + cov_loss * self.varcov_regularizer.vcr_cov_weight
        )

        loss, loss_kd, loss_ce = self.criterion(
            t, outputs, targets, targets_old, return_partial_losses=True
        )
        loss += varcov_loss.mean()
        return loss, loss_kd, loss_ce

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            # total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            (
                total_loss,
                total_acc_taw,
                total_acc_tag,
                total_num,
                total_var,
                total_cov,
                total_corr,
                total_layers_var,
                total_layers_cov,
                total_layers_corr,
            ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            self.model.eval()
            if self.model_old is not None:
                self.model_old.eval()

            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                # Forward current model

                # outputs = self.model(images)
                # loss = self.criterion(t, outputs, targets, targets_old)
                var_loss, cov_loss, corrs, feats = self.varcov_regularizer(
                    self.model.model, images, t, compute_corr=True
                )
                outputs = [head(feats) for head in self.model.heads]
                # varcov_loss = var_loss + cov_loss

                loss, loss_kd, loss_ce = self.criterion(
                    t, outputs, targets, targets_old, return_partial_losses=True
                )
                # loss += varcov_loss.mean()
                # loss, loss_kd, loss_ce = self.add_varbose_lwf(
                #     self.model, t, images, targets, targets_old
                # )

                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)

                total_var += var_loss.mean().item() * len(targets)
                total_cov += cov_loss.mean().item() * len(targets)
                total_corr += corrs.mean().item() * len(targets)

                total_layers_var += var_loss * len(targets)
                total_layers_cov += cov_loss * len(targets)
                total_layers_corr += corrs * len(targets)

                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

        if self.cka and t > 0 and self.training:
            _cka = cka(self.model, self.model_old, val_loader, self.device)
            self.logger.log_scalar(
                task=None, iter=None, name=f"t_{t}", group=f"cka", value=_cka
            )

        return (
            total_loss / total_num,
            total_var / total_num,
            total_cov / total_num,
            total_corr / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
            (total_layers_var / total_num).cpu(),
            (total_layers_cov / total_num).cpu(),
            (total_layers_corr / total_num).cpu(),
        )

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(
        self, t, outputs, targets, outputs_old=None, return_partial_losses=False
    ):
        """Returns the loss value"""
        if t > 0 and outputs_old is not None:
            # Knowledge distillation loss for all previous tasks
            kd_outputs, kd_outputs_old = (
                torch.cat(outputs[:t], dim=1),
                torch.cat(outputs_old[:t], dim=1),
            )

            if self.mc:
                g = torch.sigmoid(kd_outputs)
                q_i = torch.sigmoid(kd_outputs_old)
                loss_kd = sum(
                    torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y])
                    for y in range(kd_outputs.shape[-1])
                )
            elif self.taskwise_kd:
                loss_kd = torch.zeros(t).to(self.device)
                for _t in range(t):
                    soft_target = torch.nn.functional.softmax(
                        outputs_old[_t] / self.T, dim=1
                    )
                    output_log = torch.nn.functional.log_softmax(
                        outputs[_t] / self.T, dim=1
                    )
                    loss_kd[_t] = torch.nn.functional.kl_div(
                        output_log, soft_target, reduction="batchmean"
                    ) * (self.T**2)
                loss_kd = loss_kd.sum()
            else:
                loss_kd = self.cross_entropy(
                    kd_outputs, kd_outputs_old, exp=1.0 / self.T
                )
        else:
            loss_kd = 0

        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            loss_ce = torch.nn.functional.cross_entropy(
                torch.cat(outputs, dim=1), targets
            )
        else:
            loss_ce = torch.nn.functional.cross_entropy(
                outputs[t], targets - self.model.task_offset[t]
            )

        if return_partial_losses:
            return self.lamb * loss_kd + loss_ce, loss_kd, loss_ce
        else:
            return self.lamb * loss_kd + loss_ce
