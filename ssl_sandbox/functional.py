import torch


def entropy(p: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(torch.log(p ** (-p)), dim=dim)
