from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Tuple, Union, Optional
import torch.nn.functional as F
import torch


class VarCovRegLossInterface(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass
class VarCovRegLoss:
    vcr_var_weight: float
    vcr_cov_weight: float
    collect_layers: Callable[[torch.nn.Module], list[torch.nn.Module]]
    delta: float = 1
    hooked_layer_names: Optional[list[str]] = None
    eps: float = 1e-4
    scale: bool = True
    n_first_task: int = -1
    _initialised: bool = False
    _hooks: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))

    def initialise_hooks(self, model):
        def scale_strategy(output):
            if self.scale:
                return output - output.mean(0)
            else:
                return output

        def hook_fn(layer_name):
            def hook(module, input, output):
                output = scale_strategy(output)
                self._hooks[layer_name] = output
                return output

            return hook

        assert len(self.hooked_layer_names) == len(self.collect_layers(model))

        for name, layer in self.collect_layers(model):
            layer.register_forward_hook(hook_fn(name))

    def __call__(self, model: torch.nn.Module, inputs: torch.Tensor, t: int):
        if not self._initialised:
            self.initialise_hooks(model)
            self._initialised = True

        feats = model(inputs)

        if self.n_first_task >= 0 and t >= self.n_first_task:
            dummy_zero = torch.zeros(len(self.hooked_layer_names))
            return dummy_zero, dummy_zero, feats

        variances = []
        covariances = []

        for hook in self._hooks.values():
            v, c = self.regularize_step(hook)
            variances.append(v)
            covariances.append(c)

        variances_t = torch.stack(variances)
        covariances_t = torch.stack(covariances)

        return (
            variances_t,
            covariances_t,
            feats,
        )

    def smooth_l1(self, x, delta):
        abs_x = torch.abs(x)
        return torch.where(abs_x <= delta, x**2, 2 * delta * abs_x - delta**2)

    def regularize_step(self, feats):
        if not self.scale:
            feats = feats - feats.mean(0)
        if len(feats.shape) > 2:
            feats = feats.permute(0, 3, 2, 1)
        flattened_input = feats.flatten(start_dim=0, end_dim=-2)
        n, d = flattened_input.shape

        C = (flattened_input.T @ flattened_input) / (n - 1)
        v = torch.mean(F.relu(1 - torch.sqrt(C.diag() + self.eps)))
        c = self.smooth_l1(C.fill_diagonal_(0), self.delta).sum() / (d * (d - 1))

        return v, c


@dataclass
class NullVarCovRegLoss:
    vcr_var_weight: float = 0.0
    vcr_cov_weight: float = 0.0
    scale: bool = False
    hooked_layer_names: Optional[list[str]] = None
    _dummy_zero = torch.tensor(0.0)

    @property
    def scale_strategy(self):
        if self.scale:
            return lambda feats: feats - feats.mean(0)
        else:
            return lambda feats: feats

    def __call__(self, model, inputs, t):
        feats = model(inputs)

        return self._dummy_zero, self._dummy_zero, self.scale_strategy(feats)
