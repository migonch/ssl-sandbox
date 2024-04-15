from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import EncoderArchitecture, encoder
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import entropy


class SwAV(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            prototype_dim: int,
            num_prototypes: int,
            temp: float,
            sharpen_temp: float,
            num_sinkhorn_iters: int,
            sinkhorn_queue_size: int,
            memax_weight: float,
            dispersion_weight: float,
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
        self.sharpen_temp = sharpen_temp
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.sinkhorn_queue_size = sinkhorn_queue_size
        self.memax_weight = memax_weight
        self.dispersion_weight = dispersion_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch

    def on_fit_start(self) -> None:
        queue_size = self.sinkhorn_queue_size // self.trainer.world_size
        self.sinkhorn_queue_1 = torch.zeros(queue_size, self.num_prototypes, device=self.device)
        self.sinkhorn_queue_2 = torch.zeros(queue_size, self.num_prototypes, device=self.device)

    def to_logits(self, images):
        embeds = F.normalize(self.mlp(self.encoder(images)), dim=1)  # (n, pd)
        prototypes = F.normalize(self.prototypes, dim=1)  # (np, pd)
        return torch.matmul(embeds, prototypes.T) / self.temp  # (n, np)

    def training_step(self, batch, batch_idx):
        (_, x_1, x_2), _ = batch

        logits_1 = self.to_logits(x_1)
        logits_2 = self.to_logits(x_2)
        
        targets_1 = torch.softmax(logits_1.detach() / self.sharpen_temp, dim=1)
        targets_2 = torch.softmax(logits_2.detach() / self.sharpen_temp, dim=1)

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
        self.log(f'pretrain/bootstrap_loss', bootstrap_loss, on_step=True, on_epoch=True)

        probas_1 = torch.softmax(logits_1, dim=1)  # (batch_size, num_prototypes)
        probas_2 = torch.softmax(logits_2, dim=1)

        probas_1 = self.all_gather(probas_1, sync_grads=True)  # (world_size, batch_size, num_prototypes)
        probas_2 = self.all_gather(probas_2, sync_grads=True)

        memax_reg_1 = math.log(self.num_prototypes) - entropy(probas_1.mean(dim=(0, 1)), dim=0)
        memax_reg_2 = math.log(self.num_prototypes) - entropy(probas_2.mean(dim=(0, 1)), dim=0)
        memax_reg = (memax_reg_1 + memax_reg_2) / 2
        self.log(f'pretrain/memax_reg', memax_reg, on_step=True, on_epoch=True)

        prototypes = F.normalize(self.prototypes, dim=-1)  # (np, pd)
        logits = prototypes @ prototypes.T / self.temp
        logits.fill_diagonal_(float('-inf'))
        dispersion_reg = torch.logsumexp(logits, dim=1).mean()
        self.log(f'pretrain/dispersion_reg', dispersion_reg, on_step=True, on_epoch=True)

        loss = bootstrap_loss + self.memax_weight * memax_reg + self.dispersion_weight * dispersion_reg
        self.log(f'pretrain/loss', loss, on_epoch=True, sync_dist=True)

        pred_entropy = entropy(probas_1, dim=1).mean()
        self.log(f'pretrain/pred_entropy', pred_entropy, on_step=True, on_epoch=True)

        self.logger.log_metrics({'memax_weight': self.memax_weight}, self.global_step)

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
