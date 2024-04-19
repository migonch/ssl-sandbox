from typing import Any
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_semiring_einsum

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class MutualBinaryCodes(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            num_heads: int,
            head_hidden_dim: int,
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

        self.head = MLP(
            input_dim=self.repr_dim,
            hidden_dim=head_hidden_dim,
            output_dim=num_heads,
            num_hidden_layers=2
        )

        self.num_heads = num_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        (_, x, y), _ = batch
        n = len(x)

        l_x = self.head(self.encoder(x))  # (n, k)
        l_x = torch.stack([-l_x / 2.0, l_x / 2.0], dim=-1)  # (n, k, 2)
        p_x = torch.softmax(l_x, dim=-1)  # (n, k, 2)
        log_p_x = torch.log_softmax(l_x, dim=-1)

        l_y = self.head(self.encoder(y))
        l_y = torch.stack([-l_y / 2.0, l_y / 2.0], dim=-1)  # (n, k, 2)
        log_p_y = torch.log_softmax(l_y, dim=-1)  # (n, k, 2)

        priors = torch.einsum('kni,nlj->klij', p_x.transpose(0, 1), p_x).flatten(-2, -1).div(n)  # (k, k, 4)
        logmatmulexp = torch_semiring_einsum.compile_equation('kni,nlj->klij')
        log_priors = torch_semiring_einsum.log_einsum(logmatmulexp, log_p_x.transpose(0, 1), log_p_x).flatten(-2, -1).sub(math.log(n))  # (k, k, 4)

        prior_entropy = off_diagonal(-priors.mul(log_priors).sum(dim=-1)).mean()
        self.log(f'pretrain/prior_entropy', prior_entropy, on_step=True, on_epoch=True)

        cross_entropy = -p_x.mul(log_p_y).sum(dim=-1).mean().mul(2.0)
        self.log(f'pretrain/cross_entropy', cross_entropy, on_step=True, on_epoch=True)

        loss = cross_entropy - prior_entropy
        self.log(f'pretrain/loss', loss, on_step=True, on_epoch=True)

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
