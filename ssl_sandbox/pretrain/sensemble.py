from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import AUROC, MeanMetric

from ssl_sandbox.nn.encoder import EncoderArchitecture, encoder
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import entropy, eval_mode


class Sensemble(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            dropout_rate: float = 0.5,
            drop_channel_rate: float = 0.5,
            drop_block_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            prototype_dim: int = 128,
            num_prototypes: int = 2048,
            temp: float = 0.1,
            sharpen_temp: float = 0.25,
            num_sinkhorn_iters: int = 3,
            sinkhorn_queue_size: int = 3072,
            memax_weight: float = 1.0,
            dispersion_weight: float = 1.0,
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
        self.mlp = MLP(self.embed_dim, self.embed_dim, prototype_dim,
                       num_hidden_layers=2, dropout_rate=dropout_rate)
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, prototype_dim))
        nn.init.uniform_(self.prototypes, -(1. / prototype_dim) ** 0.5, (1. / prototype_dim) ** 0.5)

        self.num_prototypes = num_prototypes
        self.temp = temp
        self.sharpen_temp = sharpen_temp
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.sinkhorn_queue_size = sinkhorn_queue_size
        self.memax_weight = memax_weight
        self.dispersion_weight = dispersion_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.ood_scores = [
            'msp', 'maxlogit', 'energy', 'entropy', 'truncated_entropy',
            'mean_msp', 'mean_entropy', 'mean_truncated_entropy', 'expected_entropy', 'bald_score',
            'mean_msp_on_views', 'mean_entropy_on_views', 'mean_truncated_entropy_on_views',
            'expected_entropy_on_views', 'bald_score_on_views'
        ]
        self.val_metrics = nn.ModuleDict()
        for k in self.ood_scores:
            self.val_metrics.update({
                f'ood_auroc_{k}': AUROC('binary'),
                f'avg_{k}_for_id_data': MeanMetric(),
                f'avg_{k}_for_ood_data': MeanMetric(),
            })

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        queue_size = self.sinkhorn_queue_size // self.trainer.world_size
        self.sinkhorn_queue_1 = torch.zeros(queue_size, self.num_prototypes, device=self.device)
        self.sinkhorn_queue_2 = torch.zeros(queue_size, self.num_prototypes, device=self.device)

    def to_logits(self, images):
        embeds = F.normalize(self.mlp(self.encoder(images)), dim=-1)  # (n, pd)
        prototypes = F.normalize(self.prototypes, dim=-1)  # (np, pd)
        return torch.matmul(embeds, prototypes.T) / self.temp  # (n, np)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        logits_1 = self.to_logits(views_1)
        logits_2 = self.to_logits(views_2)
        
        targets_1 = torch.softmax(logits_1.detach() / self.sharpen_temp, dim=-1)
        targets_2 = torch.softmax(logits_2.detach() / self.sharpen_temp, dim=-1)

        if self.num_sinkhorn_iters > 0:
            batch_size = len(targets_1)
            queue_size = len(self.sinkhorn_queue_1)
            assert queue_size >= batch_size

            # update queue
            if queue_size > batch_size:
                self.sinkhorn_queue_1[batch_size:] = self.sinkhorn_queue_1[:-batch_size].clone()
                self.sinkhorn_queue_2[batch_size:] = self.sinkhorn_queue_1[:-batch_size].clone()
            self.sinkhorn_queue_1[:batch_size] = targets_1
            self.sinkhorn_queue_2[:batch_size] = targets_2

            if batch_size * (self.global_step + 1) >= queue_size:
                # queue is full and ready for usage
                targets_1 = self.sinkhorn(self.sinkhorn_queue_1.clone())[:batch_size]  # self.sinkhorn works inplace
                targets_2 = self.sinkhorn(self.sinkhorn_queue_2.clone())[:batch_size]
            else:
                targets_1 = self.sinkhorn(targets_1)
                targets_2 = self.sinkhorn(targets_2)

        bootstrap_loss_1 = F.cross_entropy(logits_1, targets_2)
        bootstrap_loss_2 = F.cross_entropy(logits_2, targets_1)
        bootstrap_loss = (bootstrap_loss_1 + bootstrap_loss_2) / 2

        probas_1 = torch.softmax(logits_1, dim=-1)  # (batch_size, num_prototypes)
        probas_2 = torch.softmax(logits_2, dim=-1)

        probas_1 = self.all_gather(probas_1, sync_grads=True)  # (world_size, batch_size, num_prototypes)
        probas_2 = self.all_gather(probas_2, sync_grads=True)

        memax_1 = math.log(self.num_prototypes) - entropy(probas_1.mean(dim=(0, 1)), dim=-1)
        memax_2 = math.log(self.num_prototypes) - entropy(probas_2.mean(dim=(0, 1)), dim=-1)
        memax = (memax_1 + memax_2) / 2

        prototypes = F.normalize(self.prototypes, dim=-1)  # (np, pd)
        logits = prototypes @ prototypes.T / self.temp
        logits.fill_diagonal_(float('-inf'))
        dispersion = torch.logsumexp(logits, dim=1).mean()

        loss = bootstrap_loss + self.memax_weight * memax + self.dispersion_weight * dispersion

        self.log(f'train/bootstrap_loss', bootstrap_loss, on_epoch=True, sync_dist=True)
        self.log(f'train/memax_reg', memax, on_epoch=True, sync_dist=True)
        self.log(f'train/dispersion_reg', dispersion, on_epoch=True, sync_dist=True)
        self.log(f'train/loss', loss, on_epoch=True, sync_dist=True)
        self.log(f'train/entropy', entropy(probas_1, dim=-1).mean(), on_epoch=True, sync_dist=True)
        self.logger.log_metrics({'memax_weight': self.memax_weight}, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        (images, *views), labels = batch
        ood_labels = labels == -1

        ood_scores = {}
        with eval_mode(self):
            logits = self.to_logits(images)
            probas = torch.softmax(logits, dim=-1)
            ood_scores['msp'] = -probas.max(dim=-1).values
            ood_scores['maxlogit'] = -logits.max(dim=-1).values
            ood_scores['energy'] = -torch.logsumexp(logits, dim=-1)
            ood_scores['entropy'] = entropy(probas, dim=-1)
            ood_scores['truncated_entropy'] = entropy(probas, dim=-1, truncate=100)

        with eval_mode(self, enable_dropout=True):
            ensemble_probas = torch.stack([torch.softmax(self.to_logits(images), dim=-1) for _ in range(len(views))])
            (ood_scores['mean_msp'],
             ood_scores['mean_entropy'],
             ood_scores['mean_truncated_entropy'],
             ood_scores['expected_entropy'],
             ood_scores['bald_score']) = self.compute_ood_scores(ensemble_probas)

            ensemble_probas = torch.stack([torch.softmax(self.to_logits(v), dim=-1) for v in views])
            (ood_scores['mean_msp_on_views'],
             ood_scores['mean_entropy_on_views'],
             ood_scores['mean_truncated_entropy_on_views'],
             ood_scores['expected_entropy_on_views'],
             ood_scores['bald_score_on_views']) = self.compute_ood_scores(ensemble_probas)

        for k in self.ood_scores:
            self.val_metrics[f'ood_auroc_{k}'].update(ood_scores[k], ood_labels)
            self.val_metrics[f'avg_{k}_for_id_data'].update(ood_scores[k][~ood_labels])
            self.val_metrics[f'avg_{k}_for_ood_data'].update(ood_scores[k][ood_labels])

    def on_validation_epoch_end(self):
        for k, v in self.val_metrics.items():
            self.log(f'val/{k}', v.compute(), sync_dist=True)
            v.reset()

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
        mean_probas = ensemble_probas.mean(dim=0)
        mean_msp = -mean_probas.max(dim=-1).values
        mean_entropies = entropy(mean_probas, dim=-1)
        mean_truncated_entropy = entropy(mean_probas, dim=-1, truncate=100)
        expected_entropies = entropy(ensemble_probas, dim=-1).mean(dim=0)
        bald_scores = mean_entropies - expected_entropies
        return mean_msp, mean_entropies, mean_truncated_entropy, expected_entropies, bald_scores
