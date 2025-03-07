import itertools
from argparse import ArgumentParser
from typing import Optional

import omegaconf
import torch

from src.datasets.exemplars_dataset import ExemplarsDataset
from src.loggers.exp_logger import ExperimentLogger
from src.regularizers import VarCovRegLossInterface

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Memory Aware Synapses (MAS) approach (global version)
    described in https://arxiv.org/abs/1711.09601
    Original code available at https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
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

        self.lamb = cfg.approach.kwargs.lamb
        self.alpha = cfg.approach.kwargs.alpha
        self.num_samples = cfg.approach.kwargs.fi_num_samples

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {
            n: p.clone().detach()
            for n, p in feat_ext.named_parameters()
            if p.requires_grad
        }
        # Store fisher information weight importance
        self.importance = {
            n: torch.zeros(p.shape).to(self.device)
            for n, p in feat_ext.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def set_methods_defaults(self, cfg: omegaconf.DictConfig):
        defaults = {
            "lamb": 1,
            "aplha": 0.5,
            "fi_num_samples": -1,
        }
        if not cfg.kwargs:
            cfg.kwargs = omegaconf.DictConfig(defaults)
        else:
            for key in defaults.keys():
                cfg.kwargs.setdefault(key, defaults[key])

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(
                self.model.heads[-1].parameters()
            )
        else:
            params = self.model.parameters()
        return self.optimizer_factory(params)

    # Section 4.1: MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, trn_loader):
        # Initialize importance matrices
        importance = {
            n: torch.zeros(p.shape).to(self.device)
            for n, p in self.model.model.named_parameters()
            if p.requires_grad
        }
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (
            (self.num_samples // trn_loader.batch_size + 1)
            if self.num_samples > 0
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        )
        # Do forward and backward pass to accumulate L2-loss gradients
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
            outputs = self.model.forward(images.to(self.device))
            # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
            loss = torch.norm(torch.cat(outputs, dim=1), p=2, dim=1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)
        # Eq. 2: divide by N total number of samples
        n_samples = n_samples_batches * trn_loader.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance

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

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params = {
            n: p.clone().detach()
            for n, p in self.model.model.named_parameters()
            if p.requires_grad
        }

        # calculate Fisher information
        curr_importance = self.estimate_parameter_importance(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.importance.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(
                    self.device
                )
                self.importance[n] = (
                    alpha * self.importance[n] + (1 - alpha) * curr_importance[n]
                )
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance[n] = (
                    self.alpha * self.importance[n]
                    + (1 - self.alpha) * curr_importance[n]
                )

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            # Eq. 3: memory aware synapses regularizer penalty
            for n, p in self.model.model.named_parameters():
                if n in self.importance.keys():
                    loss_reg += (
                        torch.sum(
                            self.importance[n] * (p - self.older_params[n]).pow(2)
                        )
                        / 2
                    )
            loss += self.lamb * loss_reg
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(
                torch.cat(outputs, dim=1), targets
            )
        return loss + torch.nn.functional.cross_entropy(
            outputs[t], targets - self.model.task_offset[t]
        )
