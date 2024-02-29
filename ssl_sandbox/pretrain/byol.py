# TODO: not ready yet

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import entropy, eval_mode


class BYOL(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            proj_dim: int = 256,
            temp: float = 0.1,
            teacher_temp: float = 0.025,
            memax_reg_weight: float = 1.0,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 100,
            initial_tau: float = 0.996
    ):
        super().__init__()

        self.encoder = encoder
        self.teacher = deepcopy(encoder)
        self.projector = MLP(embed_dim, embed_dim, proj_dim, num_hidden_layers=2, bias=False)
        self.predictor = MLP()

        self.num_prototypes = num_prototypes
        self.temp = temp
        self.teacher_temp = teacher_temp
        self.memax_reg_weight = memax_reg_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.initial_tau = self.tau = initial_tau

    def forward(self, images):
        return torch.softmax(self.projector(self.encoder(images)) / self.temp, dim=-1)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        logits = self.projector(self.encoder(views_1)) / self.temp  # (batch_size, num_prototypes)

        with torch.no_grad(), eval_mode(self.teacher):
            target = torch.softmax(self.projector(self.teacher(views_2)) / self.teacher_temp, dim=-1)

        bootstrap_loss = F.cross_entropy(logits, target)

        probas = torch.softmax(logits, dim=-1)
        memax_reg = math.log(self.num_prototypes) - entropy(probas.mean(dim=0), dim=-1)

        loss = bootstrap_loss + self.memax_reg_weight * memax_reg

        self.log(f'pretrain/bootstrap_loss', bootstrap_loss, on_epoch=True)
        self.log(f'pretrain/memax_reg', memax_reg, on_epoch=True)
        self.log(f'pretrain/loss', loss, on_epoch=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update teacher params
        for p, teacher_p in zip(self.encoder.parameters(), self.teacher.parameters()):
            teacher_p.data = self.tau * teacher_p.data + (1.0 - self.tau) * p.data

        # update tau
        max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        self.tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * self.global_step / max_steps) + 1) / 2

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
