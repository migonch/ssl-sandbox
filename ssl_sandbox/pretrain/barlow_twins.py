from typing import Any

import torch
import torch.nn as nn

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class BarlowTwins(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            expand_dim: int,
            lmbd: float,
            lr: float,
            weight_decay: float,
            epochs: int,
            warmup_epochs: int,
            batches_per_epoch: int,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, self.repr_dim = encoder(encoder_architecture)
        self.expander = nn.Sequential(
            MLP(
                input_dim=self.repr_dim,
                hidden_dim=expand_dim,
                output_dim=expand_dim,
                num_hidden_layers=2,
                bias=False
            ),
            nn.BatchNorm1d(expand_dim, affine=False)
        )
        self.lmbd = lmbd
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, x_1, x_2), _ = batch

        z_1 = self.expander(self.encoder(x_1))  # (batch_size, expand_dim)
        z_2 = self.expander(self.encoder(x_2))  # (batch_size, expand_dim)

        n, d = z_1.shape
        c = z_1.T @ z_2 / n  # (expand_dim, expand_dim)
        if self.trainer.world_size > 1:
            c = self.all_gather(c, sync_grads=True).mean(dim=0)
        on_diag = c.diagonal().add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).sum().div(d)

        loss = on_diag + self.lmbd * off_diag

        self.log(f'pretrain/on_diag', on_diag, on_epoch=True, sync_dist=True)
        self.log(f'pretrain/off_diag', off_diag, on_epoch=True, sync_dist=True)
        self.log(f'pretrain/loss', loss, on_epoch=True, sync_dist=True)

        return loss

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
