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
            expand_dim: int,
            i_weight: float,
            v_weight: float,
            c_weight: float,
            lr: float,
            weight_decay: float,
            epochs: int,
            warmup_epochs: int,
            batches_per_epoch: int,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, self.repr_dim = encoder(architecture=encoder_architecture)
        self.expander = MLP(
            input_dim=self.repr_dim,
            hidden_dim=expand_dim,
            output_dim=expand_dim,
            num_hidden_layers=2,
            bias=False
        )
        self.i_weight = i_weight
        self.v_weight = v_weight
        self.c_weight = c_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, x_1, x_2), _ = batch

        z_1 = self.expander(self.encoder(x_1))  # (batch_size, proj_dim)
        z_2 = self.expander(self.encoder(x_2))  # (batch_size, proj_dim)

        i_reg = F.mse_loss(z_1, z_2)
        self.log(f'pretrain/i_reg', i_reg, on_step=True, on_epoch=True)

        if self.trainer.world_size > 1:
            z_1 = self.all_gather(z_1, sync_grads=True).flatten(0, 1)
            z_2 = self.all_gather(z_2, sync_grads=True).flatten(0, 1)

        v_reg_1, c_reg_1 = unbiased_vc_reg(z_1)
        v_reg_2, c_reg_2 = unbiased_vc_reg(z_2)
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log('pretrain/unbiased_v_reg', v_reg, on_step=True, on_epoch=True)

        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'pretrain/unbiased_c_reg', c_reg, on_step=True, on_epoch=True)

        vic_reg = (
            self.i_weight * i_reg
            + self.v_weight * v_reg
            + self.c_weight * c_reg
        )
        self.log(f'pretrain/unbiased_vic_reg', vic_reg, on_step=True, on_epoch=True)

        return vic_reg

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=self.batches_per_epoch,
            pct_start=self.warmup_epochs / self.epochs,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
