from typing import Any, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import AUROC, MeanMetric

from ssl_sandbox.nn.resnet import resnet18, resnet50, adapt_to_cifar10
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import entropy, eval_mode


class Sensemble(pl.LightningModule):
    def __init__(
            self,
            encoder_architeture: Literal['resnet18', 'resnet18_cifar10', 'resnet50', 'resnet50_cifar10'],
            dropout_rate: float = 0.0,
            drop_channel_rate: float = 0.5,
            drop_block_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            prototype_dim: int = 128,
            num_prototypes: int = 2048,
            temp: float = 0.1,
            sharpen_temp: float = 0.25,
            num_sinkhorn_iters: int = 3,
            sinkhorn_queue_size: int = 2048,
            memax_weight: float = 1.0,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            **hparams: Any
    ):
        super().__init__()

        dropout_params = dict(
            drop_channel_rate=drop_channel_rate,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate
        )
        if encoder_architeture in ['resnet18', 'resnet18_cifar10']:
            encoder = resnet18(**dropout_params)
            encoder.fc = nn.Identity()
            embed_dim = 512
        elif encoder_architeture in ['resnet50', 'resnet50_cifar10']:
            encoder = resnet50(**dropout_params)
            encoder.fc = nn.Identity()
            embed_dim = 2048
        else:
            raise ValueError(f'``encoder={encoder}`` is not supported')

        if encoder_architeture in ['resnet18_cifar10', 'resnet50_cifar10']:
            encoder = adapt_to_cifar10(encoder)

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.mlp = MLP(embed_dim, embed_dim, prototype_dim, dropout_rate=dropout_rate)
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, prototype_dim))
        nn.init.uniform_(self.prototypes, -(1. / prototype_dim) ** 0.5, (1. / prototype_dim) ** 0.5)

        self.num_prototypes = num_prototypes
        self.temp = temp
        self.sharpen_temp = sharpen_temp
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.sinkhorn_queue_size = sinkhorn_queue_size
        self.memax_weight = memax_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.val_auroc_entropy = AUROC('binary')
        self.val_avg_entropy_for_ood_data = MeanMetric()
        self.val_avg_entropy_for_id_data = MeanMetric()
        self.val_auroc_mean_entropy = AUROC('binary')
        self.val_auroc_expected_entropy = AUROC('binary')
        self.val_auroc_bald_score = AUROC('binary')
        self.val_auroc_mean_entropy_on_views = AUROC('binary')
        self.val_auroc_expected_entropy_on_views = AUROC('binary')
        self.val_auroc_bald_score_on_views = AUROC('binary')

        self.save_hyperparameters()
    
    def on_fit_start(self) -> None:
        queue_size = self.sinkhorn_queue_size // self.trainer.world_size
        self.sinkhorn_queue = torch.zeros(queue_size, self.num_prototypes, device=self.device)

    def to_logits(self, images):
        embeds = F.normalize(self.mlp(self.encoder(images)), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return torch.matmul(embeds, prototypes.T) / self.temp

    def forward(self, images):
        return torch.softmax(self.to_logits(images), dim=-1)

    def training_step(self, batch, batch_idx):
        (_, student_views, teacher_views), _ = batch

        logits = self.to_logits(student_views)

        with torch.no_grad():
            targets = torch.softmax(self.to_logits(teacher_views) / self.sharpen_temp, dim=-1)

        if self.num_sinkhorn_iters > 0:
            batch_size = len(targets)
            queue_size = len(self.sinkhorn_queue)
            assert queue_size >= batch_size

            # update queue
            if queue_size > batch_size:
                self.sinkhorn_queue[batch_size:] = self.sinkhorn_queue[:-batch_size].clone()
            self.sinkhorn_queue[:batch_size] = targets

            if batch_size * (self.global_step + 1) >= queue_size:
                # queue is full and ready for usage
                targets = self.sinkhorn(self.sinkhorn_queue.clone())[:batch_size]  # self.sinkhorn works inplace
            else:
                targets = self.sinkhorn(targets)

        bootstrap_loss = F.cross_entropy(logits, targets)

        probas = torch.softmax(logits, dim=-1)  # (batch_size, num_prototypes)
        probas = self.all_gather(probas, sync_grads=True)  # (world_size, batch_size, num_prototypes)
        memax = math.log(self.num_prototypes) - entropy(probas.mean(dim=(0, 1)), dim=-1)

        loss = bootstrap_loss + self.memax_weight * memax

        self.log(f'train/bootstrap_loss', bootstrap_loss, on_epoch=True, sync_dist=True)
        self.log(f'train/memax_reg', memax, on_epoch=True, sync_dist=True)
        self.log(f'train/loss', loss, on_epoch=True, sync_dist=True)
        self.log(f'train/entropy', entropy(probas, dim=-1).mean(), on_epoch=True, sync_dist=True)
        self.logger.log_metrics({'memax_weight': self.memax_weight}, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        (images, *views), labels = batch
        ood_labels = labels == -1

        with eval_mode(self):
            entropies = entropy(self.forward(images), dim=-1)
            self.val_auroc_entropy.update(entropies, ood_labels)
            self.val_avg_entropy_for_ood_data.update(entropies[ood_labels])
            self.val_avg_entropy_for_id_data.update(entropies[~ood_labels])

        with eval_mode(self, enable_dropout=True):
            ensemble_probas = torch.stack([self.forward(images) for _ in range(len(views))])
            mean_entropies, expected_entropies, bald_scores = self.compute_ood_scores(ensemble_probas)
            self.val_auroc_mean_entropy.update(mean_entropies, ood_labels)
            self.val_auroc_expected_entropy.update(expected_entropies, ood_labels)
            self.val_auroc_bald_score.update(bald_scores, ood_labels)

            ensemble_probas = torch.stack([self.forward(v) for v in views])
            mean_entropies_on_views, expected_entropies_on_views, bald_scores_on_views = \
                self.compute_ood_scores(ensemble_probas)
            self.val_auroc_mean_entropy_on_views.update(mean_entropies_on_views, ood_labels)
            self.val_auroc_expected_entropy_on_views.update(expected_entropies_on_views, ood_labels)
            self.val_auroc_bald_score_on_views.update(bald_scores_on_views, ood_labels)

    def on_validation_epoch_end(self):
        self.log(f'val/ood_auroc_entropy', self.val_auroc_entropy.compute(), sync_dist=True)
        self.val_auroc_entropy.reset()
        self.log(f'val/avg_entropy_for_ood_data', self.val_avg_entropy_for_ood_data.compute(), sync_dist=True)
        self.val_avg_entropy_for_ood_data.reset()
        self.log(f'val/avg_entropy_for_id_data', self.val_avg_entropy_for_id_data.compute(), sync_dist=True)
        self.val_avg_entropy_for_id_data.reset()
        self.log(f'val/ood_auroc_mean_entropy', self.val_auroc_mean_entropy.compute(), sync_dist=True)
        self.val_auroc_mean_entropy.reset()
        self.log(f'val/ood_auroc_expected_entropy', self.val_auroc_expected_entropy.compute(), sync_dist=True)
        self.val_auroc_expected_entropy.reset()
        self.log(f'val/ood_auroc_bald_score', self.val_auroc_bald_score.compute(), sync_dist=True)
        self.val_auroc_bald_score.reset()
        self.log(f'val/ood_auroc_mean_entropy_on_views', self.val_auroc_mean_entropy_on_views.compute(), sync_dist=True)
        self.val_auroc_mean_entropy_on_views.reset()
        self.log(f'val/ood_auroc_expected_entropy_on_views', self.val_auroc_expected_entropy_on_views.compute(), sync_dist=True)
        self.val_auroc_expected_entropy_on_views.reset()
        self.log(f'val/ood_auroc_bald_score_on_views', self.val_auroc_bald_score_on_views.compute(), sync_dist=True)
        self.val_auroc_bald_score_on_views.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @torch.no_grad()
    def sinkhorn(self, targets: torch.Tensor) -> torch.Tensor:
        gathered_targets = self.all_gather(targets)
        world_size, batch_size, num_prototypes = gathered_targets.shape
        targets = targets / gathered_targets.sum()

        for _ in range(self.num_sinkhorn_iters):
            targets /= self.all_gather(targets).sum(dim=(0, 1))
            targets /= num_prototypes

            targets /= targets.sum(dim=-1, keepdim=True)
            targets /= world_size * batch_size

        targets *= world_size * batch_size
        return targets

    @staticmethod
    def compute_ood_scores(ensemble_probas: torch.Tensor) -> torch.Tensor:
        mean_entropies = entropy(ensemble_probas.mean(dim=0), dim=-1)
        expected_entropies = entropy(ensemble_probas, dim=-1).mean(dim=0)
        bald_scores = mean_entropies - expected_entropies
        return mean_entropies, expected_entropies, bald_scores
