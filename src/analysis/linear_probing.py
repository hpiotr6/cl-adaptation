import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

import wandb
from src.analysis import core
from src.datasets.memory_dataset import MemoryDataset


class LinearProbingModel(pl.LightningModule):
    def __init__(self, base_model, num_classes):
        super(LinearProbingModel, self).__init__()
        self.base_model = base_model

        self._freeze_extractor()

        features_size = 768  # FIXME

        self.linear_layer = nn.Linear(features_size, num_classes)
        torch.nn.init.zeros_(self.linear_layer.weight)
        torch.nn.init.zeros_(self.linear_layer.bias)

        self.save_hyperparameters(ignore=["base_model"])

    def _freeze_extractor(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        out = self.linear_layer(features)
        return out

    def configure_optimizers(self):
        return AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,
        )

    def shared_step(self, batch, mode="train"):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("%s_loss" % mode, loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("%s_acc" % mode, acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self.shared_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, mode="test")


def offset_labels(data: MemoryDataset, num_classes, data_idx):
    offset = num_classes * data_idx
    data_cp = deepcopy(data)
    data_cp.labels = list(map(lambda x: x - offset, data.labels))
    assert min(data_cp.labels) == 0
    assert max(data_cp.labels) >= num_classes - 1
    return data_cp


def init_wandb(data_task, cfg, task_idx, exp_name):
    today_date = datetime.today().strftime(r"%d-%m-%Y")
    project_name = f"probing-{today_date}"
    wandb_logger = WandbLogger(project=project_name, group=exp_name)
    task_info = {"task": task_idx, "data_task": data_task}
    wandb_logger.experiment.config.update({**cfg, **task_info})
    return wandb_logger


if __name__ == "__main__":
    expname_path = {
        "convnext_finetuning_reg": "results/2024/04.24/13-34-44/0/cifar100_fixed_finetuning",
        "convnext_finetuning_noreg": "results/2024/04.17/18-12-39/0/cifar100_fixed_finetuning_final_checkpoint",
        "convnext_ewc_reg": "results/2024/04.24/13-34-32/4/cifar100_fixed_ewc",
        "convnext_ewc_noreg": "results/2024/04.22/23-58-48/0/cifar100_fixed_ewc_final_checkpoint",
        "convnext_lwf_reg": "results/2024/04.24/13-35-04/0/cifar100_fixed_lwf",
        "convnext_lwf_noreg": "results/2024/04.27/10-23-48/0/cifar100_fixed_lwf",
        "convnext_replay_reg": "results/2024/04.24/13-34-32/2/cifar100_fixed_finetuning",
        "convnext_replay_noreg": "results/2024/04.17/18-12-39/1/cifar100_fixed_finetuning_final_checkpoint",
    }
    MAX_EPOCHS = 30
    root = Path("linear_checkpoints")
    root.mkdir(exist_ok=True)

    num_classes = 20
    device = "cuda"
    data_task = 0

    for exp_name, exp_path in expname_path.items():
        cfg = core.create_cfg(exp_path)
        cfg.data.use_test_as_val = False

        data_factory = core.DataFactory(cfg)
        train_loader, val_loader, test_loader, taskcla = data_factory[data_task]
        model_factory = core.ModelFactory(cfg, exp_path, device=device)
        for task_idx, ckpt in enumerate(sorted(model_factory.ckpts)):
            wandb_logger = init_wandb(data_task, cfg, task_idx, exp_name)
            model = model_factory.create_model(task=task_idx, num_classes=num_classes)
            linear_probing_model = LinearProbingModel(model.model, num_classes)

            ckpt_filename = (
                f"{exp_name}__dt:{data_task}_t:{task_idx}"
                + "_linear_p_best_{epoch:02d}-{val_loss:.2f}"
            )

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",  # Metric to monitor for improvement
                mode="min",  # 'min' if the metric should be minimized, 'max' if maximized
                dirpath=root,  # Directory to save checkpoints
                filename=ckpt_filename,  # Naming pattern for saved checkpoints
            )

            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                devices=1,
                callbacks=[checkpoint_callback],  # Pass the ModelCheckpoint callback
                logger=wandb_logger,
            )  # You can adjust the number of epochs and other parameters
            trainer.fit(linear_probing_model, train_loader, val_loader)

            trainer.test(linear_probing_model, test_loader, ckpt_path="best")

            wandb.finish()
