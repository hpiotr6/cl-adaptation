import importlib
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import torch
from tqdm import tqdm

from src.networks.network import LLL_Net


def choose_task(experiment_root: str, task_num: int) -> str:
    root = Path(experiment_root)
    ckpts = list(root.rglob("*.ckpt"))
    res = list(filter(lambda x: str(task_num) in x.name, ckpts))
    assert len(res) == 1
    return str(res[0])


def load_and_clean(path: str) -> dict:
    state_dict = torch.load(path, map_location="cpu")
    state_dict = {x.replace("model.", ""): y for x, y in state_dict.items()}
    return state_dict


def load_whole_model(path: str, task: int, num_classes: int) -> torch.nn.Module:
    net = getattr(importlib.import_module(name="src.networks"), "no_last_relu")
    init_model = net(pretrained=False)
    model = LLL_Net(init_model, remove_existing_head=True)

    path = choose_task(path, task)
    for _ in range(task + 1):
        model.add_head(num_classes)

    state = torch.load(path, map_location="cpu")
    info = model.load_state_dict(state, strict=False)
    pprint(info)
    return model


def load_extractor(path: str, task: int, load_and_clean) -> torch.nn.Module:
    net = getattr(importlib.import_module(name="src.networks"), "no_last_relu")
    init_model = net(pretrained=False)
    model = LLL_Net(init_model, remove_existing_head=True).model

    path = choose_task(path, task)
    state = load_and_clean(path)
    info = model.load_state_dict(state, strict=False)
    pprint(info)
    return model


@torch.no_grad()
def get_activations(model, dataloader) -> dict:
    # outputs = []
    outputs = defaultdict(list)
    for batch in tqdm(dataloader):
        inputs, labels = batch
        output = model(inputs)
        for k, v in zip(labels, output):
            outputs[k.item()].append(v)

    # return map(lambda x: torch.stack(x), outputs.values())
    return {label: torch.stack(outs) for label, outs in outputs.items()}

    # return torch.stack(outputs)
