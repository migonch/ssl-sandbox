from typing import Any

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


def covariance_matrix(embeds: torch.Tensor) -> torch.Tensor:
    """

    Args:
        embeds (torch.Tensor):
            Batch of embeddings. Size (n, d).

    Returns:
        torch.Tensor: matrix of size (d, d).
    """
    n, _ = embeds.shape
    embeds = embeds - embeds.mean(dim=0)
    return (embeds.T @ embeds).div(n - 1)


def unbiased_vc_reg(embeds: torch.Tensor) -> torch.Tensor:
    n, d = embeds.shape
    if n < 4:
        raise ValueError('Batch size must be at least 4')

    cov_1 = covariance_matrix(embeds[:n // 2])
    cov_2 = covariance_matrix(embeds[n // 2:])
    v_reg = torch.mean(cov_1.diagonal().add(-1) * cov_2.diagonal().add(-1))
    c_reg = torch.sum(off_diagonal(cov_1) * off_diagonal(cov_2)).div(d)

    return v_reg, c_reg


class UnbiasedVICReg(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            proj_dim: int = 8192,
            i_weight: float = 25.0,
            v_weight: float = 25.0,
            c_weight: float = 1.0,
            lr: float = 3e-4,
            weight_decay: float = 0.0,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, self.embed_dim = encoder(architecture=encoder_architecture)
        self.projector = MLP(
            input_dim=self.embed_dim,
            hidden_dim=proj_dim,
            output_dim=proj_dim,
            num_hidden_layers=2,
            bias=False
        )
        self.i_weight = i_weight
        self.v_weight = v_weight
        self.c_weight = c_weight
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        embeds_1 = self.projector(self.encoder(views_1))  # (batch_size, proj_dim)
        embeds_2 = self.projector(self.encoder(views_2))  # (batch_size, proj_dim)

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'pretrain/i_reg', i_reg, on_step=True, on_epoch=True)

        v_reg_1, c_reg_1 = unbiased_vc_reg(embeds_1)
        v_reg_2, c_reg_2 = unbiased_vc_reg(embeds_2)
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log('pretrain/unbiased_v_reg', v_reg, on_step=True, on_epoch=True)

        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'pretrain/unbiased_c_reg', c_reg, on_step=True, on_epoch=True)

        vic_reg = (
            self.i_weight * i_reg
            + self.v_weight * v_reg
            + self.c_weight * c_reg
        )
        self.log(f'pretrain/unbiased_vic_reg', vic_reg, on_epoch=True)

        return vic_reg

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
