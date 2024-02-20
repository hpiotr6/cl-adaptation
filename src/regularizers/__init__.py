from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Tuple, Union
import torch.nn.functional as F
import torch


@dataclass
class VarCovRegLossProtocol(Protocol):
    def __call__(self, model: torch.nn.Module, inputs: torch.Tensor): ...


@dataclass
class VarCovRegLoss(VarCovRegLossProtocol):
    vcr_var_weight: float
    vcr_cov_weight: float
    eps: float = 1e-4
    layer_names_to_hook: Union[list, None] = None
    initialised: bool = False
    hooks: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))
    scale_strategy: Callable[[torch.Tensor], torch.Tensor] = lambda feats: feats

    def initialise_hooks(self, model):
        def hook_fn(name):
            def hook(module, input, output):
                self.hooks[name] = output

            return hook

        # Initialize hooks on specified layers
        for layer_name in self.layer_names_to_hook:
            layer = dict(model.named_children())[layer_name]
            self.hooks[layer_name] = layer.register_forward_hook(hook_fn(layer_name))

    def __call__(self, model: torch.nn.Module, inputs: torch.Tensor):
        if self.layer_names_to_hook is None:
            last_layer_name = list(model.named_modules())[-1][0]
            self.layer_names_to_hook = [last_layer_name]

        if not self.initialised:
            self.initialise_hooks(model)

        feats = model(inputs)

        variances = []
        covariances = []

        for hook in [*self.hooks.values()]:
            v, c = self.regularize_step(hook)
            variances.append(v)
            covariances.append(c)

        variances_t = torch.tensor(variances)
        covariances_t = torch.tensor(covariances)

        return (
            variances_t * self.vcr_var_weight,
            covariances_t * self.vcr_cov_weight,
            self.scale_strategy(feats),
        )

    def regularize_step(self, feats):
        flattened_input = feats.flatten(start_dim=0, end_dim=-2)
        n, d = flattened_input.shape
        flattened_input = flattened_input - flattened_input.mean(dim=0)
        C = (flattened_input.T @ flattened_input) / (n - 1)
        v = torch.mean(F.relu(1 - torch.sqrt(C.diag() + self.eps)))
        diag = torch.eye(d, device=flattened_input.device)
        c = C[~diag.bool()].pow_(2).sum() / (d * (d - 1))
        return v, c


@dataclass
class NullVarCovRegLoss(VarCovRegLossProtocol):
    scale_strategy: Callable[[torch.Tensor], torch.Tensor] = lambda feats: feats
    dummy_zero = torch.tensor(0.0)

    def __call__(self, model, inputs):
        feats = model(inputs)

        return self.dummy_zero, self.dummy_zero, self.scale_strategy(feats)
