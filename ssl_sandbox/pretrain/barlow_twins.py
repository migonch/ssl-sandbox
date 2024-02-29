from typing import Any

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class BarlowTwins(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            dropout_rate: float = 0.5,
            drop_channel_rate: float = 0.5,
            drop_block_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            proj_dim: int = 8192,
            lmbd: float = 5e-3,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, self.embed_dim = encoder(
            architecture=encoder_architecture,
            drop_channel_rate=drop_channel_rate,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate
        )
        self.projector = nn.Sequential(
            MLP(
                input_dim=self.embed_dim,
                hidden_dim=proj_dim,
                output_dim=proj_dim,
                num_hidden_layers=2,
                dropout_rate=dropout_rate,
                bias=False
            ),
            nn.BatchNorm1d(proj_dim, affine=False)
        )
        self.lmbd = lmbd
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch
        views = torch.cat((views_1, views_2))

        embeds = self.projector(self.encoder(views))
        embeds_1, embeds_2 = torch.chunk(embeds, 2)

        n, d = embeds_1.shape
        c = embeds_1.T @ embeds_2 / n  # (proj_dim, proj_dim)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
