from contextlib import contextmanager

import torch
import torch.nn as nn


def entropy(p: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(torch.log(p ** (-p)), dim=dim)


def generalized_entropy(p: torch.Tensor, dim: int, M: int = 100, gamma: float = 0.1) -> torch.Tensor:
    p = torch.topk(p, k=M, dim=dim).values
    return torch.sum(p ** gamma * (1 - p) ** gamma, dim=dim)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@contextmanager
def eval_mode(module: nn.Module, enable_dropout: bool = False):
    """Copypasted from pl_bolts.callbacks.ssl_online.set_training
    """
    original_mode = module.training

    try:
        module.eval()
        if enable_dropout:
            for m in module.modules():
                if m.__class__.__name__.startswith('Drop'):
                    m.train()
        yield module
    finally:
        module.train(original_mode)
