from typing import Literal, Tuple

import torch.nn as nn

from .resnet import resnet18, resnet50, adapt_to_cifar10


EncoderArchitecture = Literal['resnet18', 'resnet18_cifar10', 'resnet50', 'resnet50_cifar10']


def encoder(architecture: EncoderArchitecture, **kwargs) -> Tuple[nn.Module, int]:
    if architecture in ['resnet18', 'resnet18_cifar10']:
        encoder = resnet18(**kwargs)
        encoder.fc = nn.Identity()
        embed_dim = 512
    elif architecture in ['resnet50', 'resnet50_cifar10']:
        encoder = resnet50(**kwargs)
        encoder.fc = nn.Identity()
        embed_dim = 2048
    else:
        raise ValueError(f'``encoder={encoder}`` is not supported')

    if architecture in ['resnet18_cifar10', 'resnet50_cifar10']:
        encoder = adapt_to_cifar10(encoder)
    
    return encoder, embed_dim
