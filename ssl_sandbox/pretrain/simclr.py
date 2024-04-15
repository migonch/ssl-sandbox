from typing import Any

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP


class SimCLR(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            proj_dim: int,
            temp: float,
            decoupled: bool,
            lr: float,
            weight_decay: float,
            epochs: int,
            warmup_epochs: int,
            batches_per_epoch: int,
            **hparams: Any  # will be dumped to yaml in logs folder
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, self.repr_dim = encoder(encoder_architecture)
        self.projector = MLP(
            input_dim=self.repr_dim,
            hidden_dim=self.repr_dim,
            output_dim=proj_dim,
            num_hidden_layers=2,
            bias=False
        )
        self.temp = temp
        self.decoupled = decoupled
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, x_1, x_2), _ = batch

        z_1 = F.normalize(self.projector(self(x_1)), dim=1)  # (batch_size, proj_dim)
        z_2 = F.normalize(self.projector(self(x_2)), dim=1)  # (batch_size, proj_dim)

        if self.trainer.world_size > 1:
            z_1 = self.all_gather(z_1, sync_grads=True).flatten(0, 1)
            z_2 = self.all_gather(z_2, sync_grads=True).flatten(0, 1)

        logits_11 = torch.matmul(z_1, z_1.T) / self.temp  # (batch_size, batch_size)
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(z_1, z_2.T) / self.temp
        pos_logits = logits_12.diag()
        if self.decoupled:
            logits_12.fill_diagonal_(float('-inf'))
        logits_22 = torch.matmul(z_2, z_2.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))

        loss_1 = torch.mean(-pos_logits + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        loss_2 = torch.mean(-pos_logits + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        loss = (loss_1 + loss_2) / 2
        self.log(f'pretrain/simclr_loss', loss, on_epoch=True)

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
