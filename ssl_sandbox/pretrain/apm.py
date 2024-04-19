from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import EncoderArchitecture, encoder
from ssl_sandbox.nn.blocks import MLP


class AdversarialPredictibilityMinimization(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            prototype_dim: int,
            num_prototypes: int,
            temp: float,
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
        self.mlp = MLP(
            input_dim=self.repr_dim,
            hidden_dim=self.repr_dim,
            output_dim=prototype_dim,
            num_hidden_layers=2,
            bias=False
        )
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, prototype_dim))
        nn.init.uniform_(self.prototypes, -(1. / prototype_dim) ** 0.5, (1. / prototype_dim) ** 0.5)

        self.num_prototypes = num_prototypes
        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def forward(self, images):
        return self.encoder(images)

    def to_logits(self, images):
        embeds = F.normalize(self.mlp(self.encoder(images)), dim=1)  # (batch_size, prototype_dim)
        prototypes = F.normalize(self.prototypes, dim=1)  # (batch_size, prototype_dim)
        return torch.matmul(embeds, prototypes.T) / self.temp  # (batch_size, num_prototypes)

    def training_step(self, batch, batch_idx):
        (_, x, y), _ = batch
        n = len(x)  # batch_size

        l_x = self.to_logits(x)  # (batch_size, num_prototypes)
        p_x = torch.softmax(l_x, dim=1)
        log_p_x = torch.log_softmax(l_x, dim=1)
        l_y = self.to_logits(y)
        p_y = torch.softmax(l_y, dim=1)
        log_p_y = torch.log_softmax(l_y, dim=1)

        priors = torch.mean(p_x, dim=0)
        log_priors = torch.logsumexp(log_p_x, dim=0) - math.log(n)  # (num_prototypes,)

        prior_entropy = -priors.mul(log_priors).sum()
        self.log(f'pretrain/prior_entropy', prior_entropy, on_step=True, on_epoch=True)

        cross_entropy = -p_x.mul(log_p_y).sum(dim=1).mean()
        self.log(f'pretrain/cross_entropy', cross_entropy, on_step=True, on_epoch=True)

        loss = cross_entropy - prior_entropy
        self.log(f'pretrain/loss', loss, on_step=True, on_epoch=True)

        p_x_entropy = -p_x.mul(log_p_x).sum(dim=1).mean()
        self.log(f'pretrain/p_x_entropy', p_x_entropy, on_step=True, on_epoch=True)

        p_y_entropy = -p_y.mul(log_p_y).sum(dim=1).mean()
        self.log(f'pretrain/p_y_entropy', p_y_entropy, on_step=True, on_epoch=True)

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
