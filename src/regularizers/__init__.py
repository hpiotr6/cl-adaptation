from dataclasses import dataclass
from typing import Protocol, Tuple
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

    def __call__(self, model: torch.nn.Module, inputs: torch.Tensor):
        feats = model(inputs)

        assert len(feats.shape) == 2

        n, d = feats.shape
        feats = feats - feats.mean(dim=0)
        C = (feats.T @ feats) / (n - 1)
        v = torch.mean(F.relu(1 - torch.sqrt(C.diag() + self.eps)))
        diag = torch.eye(d, device=feats.device)
        c = C[~diag.bool()].pow_(2).sum() / (d * (d - 1))

        return self.vcr_var_weight * v + self.vcr_cov_weight * c, feats


@dataclass
class NullVarCovRegLoss(VarCovRegLossProtocol):
    dummy_zero = torch.tensor(0.0)

    def __call__(self, model, inputs):
        feats = model(inputs)

        return self.dummy_zero, feats
