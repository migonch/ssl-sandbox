from typing import Any, Literal

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import AUROC, MeanMetric

from ssl_sandbox.nn.resnet import resnet50, adapt_to_cifar10
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal, eval_mode


class BarlowTwins(pl.LightningModule):
    def __init__(
            self,
            encoder_architeture: Literal['resnet50', 'resnet50_cifar10'],
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

        if encoder_architeture in ['resnet50', 'resnet50_cifar10']:
            encoder = resnet50(
                drop_channel_rate=drop_channel_rate,
                drop_block_rate=drop_block_rate,
                drop_path_rate=drop_path_rate
            )
            encoder.fc = nn.Identity()
            embed_dim = 2048
            if encoder_architeture == 'resnet50_cifar10':
                encoder = adapt_to_cifar10(encoder)
        else:
            raise ValueError(f'``encoder={encoder}`` is not supported')

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.projector = nn.Sequential(
            MLP(embed_dim, proj_dim, proj_dim, num_hidden_layers=2, dropout_rate=dropout_rate, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False)
        )

        self.lmbd = lmbd
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.save_hyperparameters()

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch
        views = torch.cat((views_1, views_2))

        embeds = self.projector(self.encoder(views))
        embeds_1, embeds_2 = torch.chunk(embeds, 2)

        n, d = embeds_1.shape
        c = embeds_1.T @ embeds_2 / n  # (proj_dim, proj_dim)
        c = self.all_gather(c, sync_grads=True).mean(dim=0)
        on_diag = c.diagonal().add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).sum().div(d)

        loss = on_diag + self.lmbd * off_diag

        self.log(f'train/on_diag', on_diag, on_epoch=True, sync_dist=True)
        self.log(f'train/off_diag', off_diag, on_epoch=True, sync_dist=True)
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


class BarlowTwinsOODDetection(pl.Callback):
    def __init__(self):
        super().__init__()

        self.ood_auroc = AUROC('binary')
        self.avg_ood_score_for_ood_data = MeanMetric()
        self.avg_ood_score_for_id_data = MeanMetric()

    def on_fit_start(self, trainer, pl_module):
        self.ood_auroc.to(pl_module.device)
        self.avg_ood_score_for_ood_data.to(pl_module.device)
        self.avg_ood_score_for_id_data.to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module: BarlowTwins, outputs, batch, batch_idx, dataloader_idx=0):
        (_, *views), labels = batch
        ood_labels = labels == -1

        with eval_mode(pl_module, enable_dropout=True):
            embeds = torch.stack([pl_module.projector(pl_module.encoder(v)) for v in views])
            ood_scores = embeds.var(0).mean(-1)
            self.ood_auroc.update(ood_scores, ood_labels)
            self.avg_ood_score_for_ood_data.update(ood_scores[ood_labels])
            self.avg_ood_score_for_id_data.update(ood_scores[~ood_labels])

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.log('val/ood_auroc', self.ood_auroc.compute(), sync_dist=True)
        self.ood_auroc.reset()
        self.log(f'val/avg_ood_score_for_ood_data', self.avg_ood_score_for_ood_data.compute(), sync_dist=True)
        self.avg_ood_score_for_ood_data.reset()
        self.log(f'val/avg_ood_score_for_id_data', self.avg_ood_score_for_id_data.compute(), sync_dist=True)
        self.avg_ood_score_for_id_data.reset()
