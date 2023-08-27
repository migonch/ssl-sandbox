from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal, eval_mode


class VICReg(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            dropout_rate: float = 0.5,
            drop_channel_rate: float = 0.5,
            drop_block_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            proj_dim: int = 8192,
            i_weight: float = 25.0,
            v_weight: float = 25.0,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters(ignore='encoder')

        self.encoder, self.embed_dim = encoder(
            architecture=encoder_architecture,
            drop_channel_rate=drop_channel_rate,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate
        )
        self.projector = MLP(self.embed_dim, proj_dim, proj_dim, num_hidden_layers=2,
                             dropout_rate=dropout_rate, bias=False)

        self.i_weight = i_weight
        self.v_weight = v_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        embeds_1 = self.projector(self.encoder(views_1))  # (batch_size, proj_dim)
        embeds_2 = self.projector(self.encoder(views_2))  # (batch_size, proj_dim)

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'train/i_reg', i_reg, on_epoch=True)

        embeds_1 = embeds_1 - embeds_1.mean(dim=0)
        embeds_2 = embeds_2 - embeds_2.mean(dim=0)

        eps = 1e-4
        v_reg_1 = torch.mean(F.relu(1 - torch.sqrt(embeds_1.var(dim=0) + eps)))
        v_reg_2 = torch.mean(F.relu(1 - torch.sqrt(embeds_2.var(dim=0) + eps)))
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log(f'train/v_reg', v_reg, on_epoch=True)

        n, d = embeds_1.shape
        c_reg_1 = off_diagonal(embeds_1.T @ embeds_1).div(n - 1).pow_(2).sum().div(d)
        c_reg_2 = off_diagonal(embeds_2.T @ embeds_2).div(n - 1).pow_(2).sum().div(d)
        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'train/c_reg', c_reg, on_epoch=True)

        vic_reg = self.i_weight * i_reg + self.v_weight * v_reg + c_reg
        self.log(f'train/vic_reg', vic_reg, on_epoch=True)

        return vic_reg

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
