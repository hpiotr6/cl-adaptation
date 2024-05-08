# path -> args -> model
#              -> dataloader


from collections import defaultdict
import importlib
import os
from pathlib import Path
from pprint import pprint
import re

from tqdm import tqdm

from src.datasets.data_loader import get_loaders
from omegaconf import DictConfig, OmegaConf
import torch

from src.networks.network import LLL_Net


def create_cfg(dir_path):
    import json

    ckpt_names = []
    for file in os.listdir(os.path.join(dir_path, "models")):
        if file.endswith(".ckpt"):
            ckpt_names.append(file)
    for file in os.listdir(dir_path):
        if file.startswith("args"):
            json_file = os.path.join(dir_path, file)

    with open(json_file) as file:
        data = json.load(file)

    args = OmegaConf.create(data)

    return args


@torch.no_grad()
def get_activations(model, dataloader, device="cpu") -> dict:
    model.to(device)
    outputs = defaultdict(list)
    for batch in tqdm(dataloader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        for k, v in zip(labels, output):
            outputs[k.item()].append(v)

    return {label: torch.stack(outs) for label, outs in outputs.items()}


class ModelFactory:
    def __init__(
        self,
        cfg: OmegaConf,
        experiment_root: str,
        device="cpu",
    ) -> None:
        self.cfg = cfg
        self.experiment_root = experiment_root
        self.device = device

    @property
    def ckpts(self):
        root = Path(self.experiment_root)
        return list(root.rglob("*.ckpt"))

    def _choose_task(self, task_num: int) -> str:
        res = list(filter(lambda x: str(task_num) in x.name, self.ckpts))
        assert len(res) == 1
        return str(res[0])

    def create_model(self, task: int, num_classes: int) -> torch.nn.Module:
        net = getattr(
            importlib.import_module(name="src.networks"), self.cfg.model.network
        )
        init_model = net(pretrained=False)
        cifar_in_data = "cifar" in self.cfg.data.datasets[0]
        model = LLL_Net(init_model, is_cifar=cifar_in_data, remove_existing_head=True)

        if self.cfg.training.vcreg:
            self._initialise_hooks(model.model)

        path = self._choose_task(task)
        for _ in range(task + 1):
            model.add_head(num_classes)

        state = torch.load(path, map_location=self.device)
        info = model.load_state_dict(state, strict=False)
        pprint(info)
        return model

    def _initialise_hooks(self, model):
        def scale_strategy(output):
            return output - output.mean(0)

        def hook_fn(layer_name):
            def hook(module, input, output):
                output = scale_strategy(output)
                return output

            return hook

        for name, layer in self._collect_layers(model):
            layer.register_forward_hook(hook_fn(name))

    def _collect_layers(self, model: torch.nn.Module):
        compiled_pattern = re.compile(self.cfg.training.vcreg.reg_layers)
        matched_layers = [
            (name, module)
            for name, module in model.named_modules()
            if re.match(compiled_pattern, name)
        ]

        if not matched_layers:
            raise ValueError(
                f"No layers matching the pattern '{self.cfg.training.vcreg.reg_layers}' were found."
            )

        return matched_layers


class DataFactory:
    def __init__(
        self,
        cfg: OmegaConf,
    ) -> None:
        (
            self.trn_loader,
            self.val_loader,
            self.tst_loader,
            self.taskcla,
        ) = get_loaders(
            cfg.data.datasets,
            cfg.data.num_tasks,
            cfg.data.nc_first_task,
            cfg.data.nc_per_task,
            cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            max_classes_per_dataset=cfg.data.max_classes_per_dataset,
            max_examples_per_class_trn=cfg.data.max_examples_per_class_trn,
            max_examples_per_class_val=cfg.data.max_examples_per_class_val,
            max_examples_per_class_tst=cfg.data.max_examples_per_class_tst,
            extra_aug=cfg.data.extra_aug,
            validation=0.0 if cfg.data.use_test_as_val else 0.1,
        )

    def __getitem__(self, items):
        return (
            self.trn_loader[items],
            self.val_loader[items],
            self.tst_loader[items],
            self.taskcla[items],
        )
