from typing import Optional
import omegaconf
import torch
import itertools
from argparse import ArgumentParser

from src.datasets.exemplars_dataset import ExemplarsDataset
from src.loggers.exp_logger import ExperimentLogger
from src.regularizers import VarCovRegLossInterface
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Elastic Weight Consolidation (EWC) approach
    described in http://arxiv.org/abs/1612.00796
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
        self.sampling_type = cfg.approach.kwargs.fi_sampling_type
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
        self.fisher = {
            n: torch.zeros(p.shape).to(self.device)
            for n, p in feat_ext.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def set_methods_defaults(self, cfg: omegaconf.DictConfig):
        defaults = {
            "lamb": 5000,
            "alpha": 0.5,
            "fi_sampling_type": "max_pred",
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

    def compute_fisher_matrix_diag(self, trn_loader):
        # Store Fisher Information
        fisher = {
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
        # Do forward and backward pass to compute the fisher information
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = self.model.forward(images.to(self.device))

            if self.sampling_type == "true":
                # Use the labels to compute the gradients based on the CE-loss with the ground truth
                preds = targets.to(self.device)
            elif self.sampling_type == "max_pred":
                # Not use labels and compute the gradients related to the prediction the model has learned
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif self.sampling_type == "multinomial":
                # Use a multinomial sampling to compute the gradients
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
            self.optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        # Apply mean across all samples
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

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
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(
                    self.device
                )
                self.fisher[n] = alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
            else:
                self.fisher[n] = (
                    self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n]
                )

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.model.model.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += (
                        torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2))
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
