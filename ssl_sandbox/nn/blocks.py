import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_hidden_layers: int = 1,
            dropout_rate: float = 0.0,
            bias: bool = True
    ) -> None:
        super().__init__()

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
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
