from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Tuple, Union
import torch.nn.functional as F
import torch


@dataclass
class VarCovRegLossProtocol(Protocol):
    def __call__(
        self, model: torch.nn.Module, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        ...


@dataclass
class VarCovRegLoss(VarCovRegLossProtocol):
    vcr_var_weight: float
    vcr_cov_weight: float
    eps: float = 1e-4
    layer_names_to_hook: Union[list, None] = None
    initialised: bool = False
    hooks: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))

    # def initialise_hooks(self, model):
    #     def hook_fn(name):
    #         def hook(module, input, output):
    #             self.hooks[name] = output

    #         return hook

    #     # Initialize hooks on specified layers
    #     for layer_name in self.layer_names_to_hook:
    #         layer = dict(model.named_children())[layer_name]
    #         self.hooks[layer_name] = layer.register_forward_hook(hook_fn(layer_name))

    def __call__(self, model: torch.nn.Module, inputs: torch.Tensor):
        feats = model(inputs)
        v, c = self.regularize_step(feats)
        feats = feats - feats.mean(dim=0)
        return v * self.vcr_var_weight, c * self.vcr_cov_weight, feats
        # variance_sum = 0
        # covariance_sum = 0

        # for hook in [*self.hooks.values()]:
        #     v, c = self.regularize_step(hook)
        #     variance_sum += v
        #     covariance_sum += c

        # return (
        #     self.vcr_var_weight * variance_sum,
        #     self.vcr_cov_weight * covariance_sum,
        #     feats,
        # )

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
    dummy_zero = torch.tensor(0.0)

    def __call__(self, model, inputs):
        feats = model(inputs)
        feats = feats - feats.mean(dim=0)

        return self.dummy_zero, self.dummy_zero, feats
