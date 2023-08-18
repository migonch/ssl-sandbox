from typing import *
import json
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import eval_mode


class Probing(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            num_classes: int,
            lr: float = 3e-4
    ):
        super().__init__()

        self.save_hyperparameters(ignore='encoder')

        self.encoder = encoder
        self.linear_head = nn.Linear(embed_dim, num_classes)
        self.nonlinear_head = MLP(embed_dim, embed_dim, num_classes)

        self.num_classes = num_classes
        self.lr = lr

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        images, labels = batch

        with torch.no_grad(), eval_mode(self.encoder):
            embeds = self.encoder(images)

        for prefix in ['linear', 'nonlinear']:
            head = getattr(self, f'{prefix}_head')
            loss = F.cross_entropy(head(embeds), labels)
            self.log(f'train/{prefix}_probing_loss', loss, on_epoch=True)
            self.manual_backward(loss)

        optimizer.step()

    def on_validation_epoch_start(self) -> None:
        self.val_lin_prob_acc = Accuracy('multiclass', num_classes=self.num_classes).to(self.device)
        self.val_nonlin_prob_acc = Accuracy('multiclass', num_classes=self.num_classes).to(self.device)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images[labels != -1], labels[labels != -1]  # filter out ood examples

        embeds = self.encoder(images)

        self.val_lin_prob_acc.update(self.linear_head(embeds), labels)
        self.val_nonlin_prob_acc.update(self.nonlinear_head(embeds), labels)

    def on_validation_epoch_end(self) -> None:
        self.log(f'val/linear_probing_accuracy', self.val_lin_prob_acc.compute())
        self.log(f'val/nonlinear_probing_accuracy', self.val_nonlin_prob_acc.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class OnlineProbing(pl.Callback):
    def __init__(
            self,
            embed_dim: int,
            num_classes: int,
            lr: float = 3e-4
    ):
        super().__init__()

        self.linear_head = nn.Linear(embed_dim, num_classes)
        self.linear_optimizer = torch.optim.Adam(self.linear_head.parameters(), lr=lr)

        self.nonlinear_head = MLP(embed_dim, embed_dim, num_classes)
        self.nonlinear_optimizer = torch.optim.Adam(self.nonlinear_head.parameters(), lr=lr)

        self.val_lin_prob_acc = Accuracy('multiclass', num_classes=num_classes)
        self.val_nonlin_prob_acc = Accuracy('multiclass', num_classes=num_classes)

    def on_fit_start(self, trainer, pl_module):
        self.linear_head.to(pl_module.device)
        self.nonlinear_head.to(pl_module.device)

        self.val_lin_prob_acc.to(pl_module.device)
        self.val_nonlin_prob_acc.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        (images, *_), labels = batch

        with torch.no_grad(), eval_mode(pl_module.encoder):
            embeds = pl_module.encoder(images)

        for prefix in ['linear', 'nonlinear']:
            head = getattr(self, f'{prefix}_head')
            optimizer = getattr(self, f'{prefix}_optimizer')

            optimizer.zero_grad()
            loss = F.cross_entropy(head(embeds), labels)
            pl_module.log(f'train/{prefix}_probing_loss', loss, on_epoch=True, sync_dist=True)
            loss.backward()
            optimizer.step()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        (images, *_), labels = batch
        images, labels = images[labels != -1], labels[labels != -1]  # filter out ood examples

        with torch.no_grad(), eval_mode(pl_module.encoder):
            embeds = pl_module.encoder(images)

        self.val_lin_prob_acc.update(self.linear_head(embeds), labels)
        self.val_nonlin_prob_acc.update(self.nonlinear_head(embeds), labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log(f'val/linear_probing_accuracy', self.val_lin_prob_acc.compute(), sync_dist=True)
        self.val_lin_prob_acc.reset()
        pl_module.log(f'val/nonlinear_probing_accuracy', self.val_nonlin_prob_acc.compute(), sync_dist=True)
        self.val_nonlin_prob_acc.reset()
