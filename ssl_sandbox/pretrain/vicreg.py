from typing import Any

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class VICReg(pl.LightningModule):
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

        self.encoder, self.repr_dim = encoder(encoder_architecture)
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

        z_1 = self.expander(self.encoder(x_1))  # (batch_size, expand_dim)
        z_2 = self.expander(self.encoder(x_2))  # (batch_size, expand_dim)

        i_reg = F.mse_loss(z_1, z_2)
        self.log(f'pretrain/i_reg', i_reg, on_step=True, on_epoch=True)
        
        if self.trainer.world_size > 1:
            z_1 = self.all_gather(z_1, sync_grads=True).flatten(0, 1)
            z_2 = self.all_gather(z_2, sync_grads=True).flatten(0, 1)

        z_1 = z_1 - z_1.mean(dim=0)
        z_2 = z_2 - z_2.mean(dim=0)

        eps = 1e-4
        v_reg_1 = torch.mean(F.relu(1 - torch.sqrt(z_1.var(dim=0) + eps)))
        v_reg_2 = torch.mean(F.relu(1 - torch.sqrt(z_2.var(dim=0) + eps)))
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log(f'pretrain/v_reg', v_reg, on_step=True, on_epoch=True)

        n, d = z_1.shape
        c_reg_1 = off_diagonal(z_1.T @ z_1).div(n - 1).pow_(2).sum().div(d)
        c_reg_2 = off_diagonal(z_2.T @ z_2).div(n - 1).pow_(2).sum().div(d)
        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'pretrain/c_reg', c_reg, on_step=True, on_epoch=True)

        vic_reg = (
            self.i_weight * i_reg
            + self.v_weight * v_reg
            + self.c_weight * c_reg
        )
        self.log(f'pretrain/vic_reg', vic_reg, on_step=True, on_epoch=True)

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
