from typing import *

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class EndToEnd(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            num_classes: int,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10
    ):
        super().__init__()

        self.save_hyperparameters(ignore='encoder')

        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        loss = F.cross_entropy(self.head(self.encoder(images)), labels)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        images, labels = images[labels != -1], labels[labels != -1]  # filter out ood examples
        acc = accuracy(self.head(self.encoder(images)), labels, 'multiclass', self.num_classes)
        self.log('val/accuracy', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
