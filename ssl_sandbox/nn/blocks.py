from typing import Literal
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_hidden_layers: int = 1,
            norm: Literal['bn', 'none'] = 'bn',
            dropout_rate: float = 0.0,
            bias: bool = True
    ) -> None:
        super().__init__()
        
        assert num_hidden_layers >= 1
        
        match norm:
            case 'bn':
                norm_cls = nn.BatchNorm1d
            case 'none':
                norm_cls = nn.Identity
            case _:
                raise ValueError(norm)

        hidden_layers = [
            nn.Linear(input_dim, hidden_dim),
            norm_cls(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]
        for _ in range(num_hidden_layers - 1):
            hidden_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                norm_cls(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        self.layers = nn.ModuleList([
            *hidden_layers,
            nn.Linear(hidden_dim, output_dim, bias),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
