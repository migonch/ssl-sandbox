from typing import Any
from sklearn.metrics import roc_auc_score
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal, eval_mode


class BarlowTwins(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            proj_dim: int = 8192,
            lmbd: float = 5e-3,
            unbiased: bool = False,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters(ignore='encoder')

        self.encoder = encoder
        self.projector = nn.Sequential(
            MLP(embed_dim, proj_dim, proj_dim, num_hidden_layers=2, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False)
        )

        self.lmbd = lmbd
        self.unbiased = unbiased
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        embeds_1 = self.projector(self.encoder(views_1))  # (batch_size, proj_dim)
        embeds_2 = self.projector(self.encoder(views_2))  # (batch_size, proj_dim)
        
        n, d = embeds_1.shape
        if not self.unbiased:
            c = embeds_1.T @ embeds_2 / (n - 1)  # (proj_dim, proj_dim)
            on_diag = c.diagonal().add_(-1).pow_(2).mean()
            off_diag = off_diagonal(c).pow_(2).sum().div(d)
        else:
            assert n >= 4
            c_1 = embeds_1[:n // 2].T @ embeds_2[:n // 2] / (n // 2 - 1)
            c_2 = embeds_1[n // 2:].T @ embeds_2[n // 2:] / (n // 2 - 1)
            on_diag = torch.mean(c_1.diagonal().add_(-1) * c_2.diagonal().add_(-1))
            off_diag = torch.sum(off_diagonal(c_1) * off_diagonal(c_2)).div(d)

        loss = on_diag + self.lmbd * off_diag
        
        mse_loss = F.mse_loss(embeds_1, embeds_2)

        self.log(f'train/on_diag', on_diag, on_epoch=True)
        self.log(f'train/off_diag', off_diag, on_epoch=True)
        self.log(f'train/mse_loss', mse_loss, on_epoch=True)
        self.log(f'train/loss', loss, on_epoch=True)

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


class BarlowTwinsOODDetection(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.ood_labels = []
        self.ood_scores = []
        self.md_scores = []

    def on_validation_batch_end(self, trainer, pl_module: BarlowTwins, outputs, batch, batch_idx, dataloader_idx=0):
        (images, *views), labels = batch
        ood_labels = labels.cpu() == -1

        with eval_mode(pl_module.encoder, enable_dropout=True), eval_mode(pl_module.projector):
            embeds = torch.stack([pl_module.projector(pl_module.encoder(v)) for v in views]).detach().cpu()
            ood_scores = embeds.var(0).mean(-1)

        with eval_mode(pl_module.encoder), eval_mode(pl_module.projector):
            embeds = pl_module.projector(pl_module.encoder(images)).detach().cpu()
            md_scores = embeds.pow_(2).sum(-1)

        self.ood_labels.extend(ood_labels.tolist())
        self.ood_scores.extend(ood_scores.tolist())
        self.md_scores.extend(md_scores.tolist())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.log('val/ood_auroc', roc_auc_score(self.ood_labels, self.ood_scores))
        self.log('val/ood_auroc_md', roc_auc_score(self.ood_labels, self.md_scores))

        ood_scores = np.array(self.ood_scores)
        ood_labels = np.array(self.ood_labels)
        self.log('val/mean_ood_score_for_ood_data', ood_scores[ood_labels].mean())
        self.log('val/mean_ood_score_for_id_data', ood_scores[~ood_labels].mean())
