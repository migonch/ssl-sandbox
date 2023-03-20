import torch


def entropy(p: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(torch.log(p ** (-p)), dim=dim)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
