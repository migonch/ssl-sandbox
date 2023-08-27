from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class IBFCodes(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            dropout_rate: float = 0.5,
            drop_channel_rate: float = 0.5,
            drop_block_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            code_dim: int = 2048,
            sharpen_temp: float = 0.25,
            reg_weight: float = 1.0,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            **hparams: Any
    ):
        super().__init__()

        self.encoder, self.embed_dim = encoder(
            architecture=encoder_architecture,
            drop_channel_rate=drop_channel_rate,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate
        )
        self.projector = MLP(self.embed_dim, code_dim, code_dim, num_hidden_layers=2, dropout_rate=dropout_rate)

        self.sharpen_temp = sharpen_temp
        self.reg_weight = reg_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        (_, student_views, teacher_views), _ = batch

        logits = self.projector(self.encoder(student_views))

        with torch.no_grad():
            targets = torch.sigmoid(self.projector(self.encoder(teacher_views)) / self.sharpen_temp)

        bootstrap_loss = F.binary_cross_entropy_with_logits(logits, targets)

        n, _ = logits.shape
        probas = torch.sigmoid(logits)  # (n, k)
        p_11 = probas.T @ probas / n  # (k, k)
        p_11 = self.all_gather(p_11, sync_grads=True).mean(dim=0)
        p_1 = probas.mean(dim=0)  # (k, k)
        p_1 = self.all_gather(p_1, sync_grads=True).mean(dim=0)
        p_01 = p_1 - p_11
        p_10 = p_01.T
        p_00 = 1 - p_01 - p_10 - p_11
        pairwise_entropies = off_diagonal(
            p_00.pow(-p_00).log()
            + p_01.pow(-p_01).log()
            + p_10.pow(-p_10).log()
            + p_11.pow(-p_11).log()
        )
        reg = 2 * math.log(2) - pairwise_entropies.mean()

        loss = bootstrap_loss + self.reg_weight * reg

        self.log(f'train/bootstrap_loss', bootstrap_loss, on_epoch=True, sync_dist=True)
        self.log(f'train/reg', reg, on_epoch=True, sync_dist=True)
        self.log(f'train/loss', loss, on_epoch=True, sync_dist=True)

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
