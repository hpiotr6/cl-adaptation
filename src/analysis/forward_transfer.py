from copy import deepcopy
import os
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
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import wandb

from src.analysis import core
from src.datasets.memory_dataset import MemoryDataset


class ClassicCNN(pl.LightningModule):
    def __init__(self, base_model, num_classes, t):
        super(ClassicCNN, self).__init__()
        self.base_model = base_model

        features_size = 768  # FIXME

        self.linear_layer = nn.Linear(features_size, num_classes)
        self.t = t
        self.num_classes = num_classes

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
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[30, 60, 80],
            gamma=0.1,
        )

        return [optimizer], [scheduler]

    def shared_step(self, batch, mode="train"):
        x, y = batch
        y -= self.t * self.num_classes
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
    project_name = f"2205forward_transfer-{today_date}-test_only"
    wandb_logger = WandbLogger(project=project_name, group=exp_name)
    task_info = {"task": task_idx, "data_task": data_task}
    wandb_logger.experiment.config.update({**OmegaConf.to_container(cfg), **task_info})
    return wandb_logger


# if __name__ == "__main__":
#     MAX_EPOCHS = 100
#     root = Path("from_scratch_new")
#     root.mkdir(exist_ok=True)

#     cfg_path = "results/2024/04.24/13-34-44/0/cifar100_fixed_finetuning"
#     exp_name = "from_scratch_02"
#     num_classes = 20
#     device = "cuda"

#     task_idx = 0

#     data_tasks = range(5)
#     for data_task in data_tasks:
#         cfg = core.create_cfg(cfg_path)
#         cfg.data.use_test_as_val = False

#         data_factory = core.DataFactory(cfg)
#         train_loader, val_loader, test_loader, taskcla = data_factory[data_task]
#         model_factory = core.ModelFactory(cfg, cfg_path, device=device)
#         model = model_factory.load_model(task=0, num_classes=num_classes)

#         wandb_logger = init_wandb(data_task, cfg, task_idx, exp_name)
#         linear_probing_model = ClassicCNN(model.model, num_classes, t=data_task)

#         # ckpt_filename = (
#         #     f"{exp_name}__dt:{data_task}_t:{task_idx}"
#         #     + "_linear_p_best_{epoch:02d}-{val_loss:.2f}"
#         # )

#         # checkpoint_callback = ModelCheckpoint(
#         #     monitor="val_loss",  # Metric to monitor for improvement
#         #     mode="min",  # 'min' if the metric should be minimized, 'max' if maximized
#         #     dirpath=root,  # Directory to save checkpoints
#         #     filename=ckpt_filename,  # Naming pattern for saved checkpoints
#         # )

#         trainer = pl.Trainer(
#             max_epochs=MAX_EPOCHS,
#             devices=1,
#             logger=wandb_logger,
#         )  # You can adjust the number of epochs and other parameters
#         trainer.fit(linear_probing_model, train_loader, val_loader)

#         trainer.test(linear_probing_model, test_loader)

#         wandb.finish()


@hydra.main(version_base=None, config_path=None, config_name=None)
def my_app(cfg: DictConfig) -> None:
    MAX_EPOCHS = cfg.max_epochs

    cfg_path = "results/2024/04.24/13-34-44/0/cifar100_fixed_finetuning"
    exp_name = "convnext_from_scratch"
    num_classes = 20
    device = cfg.device

    task_idx = 0
    data_task = cfg.data_task

    config = core.create_cfg(cfg_path)

    data_factory = core.DataFactory(config)
    train_loader, _, test_loader, taskcla = data_factory[data_task]
    model_factory = core.ModelFactory(config, cfg_path, device=device)
    model = model_factory.load_model(task=0, num_classes=num_classes)

    wandb_logger = init_wandb(data_task, cfg, task_idx, exp_name)

    linear_probing_model = ClassicCNN(model.model, num_classes, t=data_task)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        logger=wandb_logger,
    )
    trainer.fit(linear_probing_model, train_loader, test_loader)

    trainer.test(linear_probing_model, test_loader)


if __name__ == "__main__":
    my_app()
