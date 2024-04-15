from typing import Any
import math

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class IBFCodes(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            code_dim: int,
            sharpen_temp: float,
            reg_weight: float,
            lr: float,
            weight_decay: float,
            epochs: int,
            warmup_epochs: int,
            batches_per_epoch: int,
            **hparams: Any
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.encoder, self.embed_dim = encoder(encoder_architecture)
        self.projector = MLP(
            input_dim=self.embed_dim,
            hidden_dim=code_dim,
            output_dim=code_dim,
            num_hidden_layers=2,
            bias=False
        )
        self.sharpen_temp = sharpen_temp
        self.reg_weight = reg_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def training_step(self, batch, batch_idx):
        (_, x_student, x_teacher), _ = batch

        logits = self.projector(self.encoder(x_student))

        with torch.no_grad():
            targets = torch.sigmoid(self.projector(self.encoder(x_teacher)) / self.sharpen_temp)

        bootstrap_loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log(f'pretrain/bootstrap_loss', bootstrap_loss, on_step=True, on_epoch=True)

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
        ibf_reg = 2 * math.log(2) - pairwise_entropies.mean()
        self.log(f'pretrain/ibf_reg', ibf_reg, on_step=True, on_epoch=True)

        loss = bootstrap_loss + self.reg_weight * ibf_reg
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
