from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F


def into_2d(feats: torch.Tensor):
    if len(feats.shape) == 2:
        return feats

    feats = feats.permute(0, 3, 2, 1)
    return feats.flatten(start_dim=0, end_dim=-2)


# def scale_strategy(feats, scale=False):
#     if scale:
#         return feats - feats.mean(0)
#     return feats


class VarCovRegLossInterface(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        t: int,
        compute_corr: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


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
        def hook_fn(layer_name):
            def hook(module, input, output):
                # output = scale_strategy(into_2d(output), self.scale)
                self._hooks[layer_name] = output
                return output

            return hook

        assert len(self.hooked_layer_names) == len(self.collect_layers(model))

        for name, layer in self.collect_layers(model):
            layer.register_forward_hook(hook_fn(name))

    def __call__(
        self, model: torch.nn.Module, inputs: torch.Tensor, t: int, compute_corr=False
    ):
        if not self._initialised:
            self.initialise_hooks(model)
            self._initialised = True

        feats = model(inputs)

        if self.n_first_task >= 0 and t >= self.n_first_task:
            dummy_zero = torch.zeros(len(self.hooked_layer_names))
            return dummy_zero, dummy_zero, feats

        variances = []
        covariances = []
        correlations = []

        for hook in self._hooks.values():
            v, c, corr = self.regularize_step(hook, compute_corr)
            variances.append(v)
            covariances.append(c)
            correlations.append(corr)

        variances_t = torch.stack(variances)
        covariances_t = torch.stack(covariances)
        correlations_t = torch.stack(correlations)

        return (
            variances_t,
            covariances_t,
            correlations_t,
            feats,
        )

    def smooth_l1(self, x, delta):
        abs_x = torch.abs(x)
        return torch.where(abs_x <= delta, x**2, 2 * delta * abs_x - delta**2)

    def regularize_step(self, feats, compute_corr):
        flattened_input = into_2d(feats)
        demeaned_flattended = flattened_input - flattened_input.mean(0)
        # demeaned_flattend = scale_strategy(flattened_input, not self.scale)
        n, d = flattened_input.shape

        C = (demeaned_flattended.T @ demeaned_flattended) / (n - 1)
        v = torch.mean(F.relu(1 - torch.sqrt(C.diag() + self.eps)))
        c = self.smooth_l1(C.fill_diagonal_(0), self.delta).sum() / (d * (d - 1))
        if compute_corr:
            corr = torch.abs(
                torch.nan_to_num(
                    torch.corrcoef(flattened_input.T).fill_diagonal_(0), nan=0.0
                )
            ).sum() / (d * (d - 1))

        else:
            corr = torch.tensor(float("nan"))

        return v, c, corr


@dataclass
class NullVarCovRegLoss:
    vcr_var_weight: float = 0.0
    vcr_cov_weight: float = 0.0
    scale: bool = False
    hooked_layer_names: Optional[list[str]] = None
    _dummy_zero = torch.tensor(0.0)

    def __call__(self, model, inputs, t):
        feats = model(inputs)
        return (
            self._dummy_zero,
            self._dummy_zero,
            self._dummy_zero,
            feats,
            # scale_strategy(into_2d(feats), self.scale),
        )
