import collections
import math
from sklearn.metrics import roc_auc_score
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.functional import entropy, eval_mode


class Projector(nn.Module):
    def __init__(self, embed_dim: int, num_prototypes: int):
        super().__init__()

        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, embed_dim))
        nn.init.uniform_(self.prototypes, -(1. / embed_dim) ** 0.5, (1. / embed_dim) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return torch.matmul(x, prototypes.T)


class Sensemble(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            num_prototypes: int = 2048,
            temp: float = 0.1,
            teacher_temp: float = 0.025,
            memax_reg_weight: float = 1.0,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            ema: bool = False,
            initial_tau: float = 0.996
    ):
        super().__init__()

        self.save_hyperparameters(ignore='encoder')

        self.encoder = encoder
        # self.teacher = encoder
        self.projector = Projector(embed_dim, num_prototypes)

        self.num_prototypes = num_prototypes
        self.temp = temp
        self.teacher_temp = teacher_temp
        self.memax_reg_weight = memax_reg_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.ema = ema
        self.initial_tau = self.tau = initial_tau

    def forward(self, images):
        return torch.softmax(self.projector(self.encoder(images)) / self.temp, dim=-1)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        logits = self.projector(self.encoder(views_1)) / self.temp  # (batch_size, num_prototypes)

        with torch.no_grad():
            target = torch.softmax(self.projector(self.encoder(views_2)) / self.teacher_temp, dim=-1)

        bootstrap_loss = F.cross_entropy(logits, target)

        probas = torch.softmax(logits, dim=-1)
        memax_reg = math.log(self.num_prototypes) - entropy(probas.mean(dim=0), dim=-1)

        loss = bootstrap_loss + self.memax_reg_weight * memax_reg

        self.log(f'train/bootstrap_loss', bootstrap_loss, on_epoch=True)
        self.log(f'train/memax_reg', memax_reg, on_epoch=True)
        self.log(f'train/loss', loss, on_epoch=True)

        return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.ema:
    #         # update teacher params
    #         for p, teacher_p in zip(self.encoder.parameters(), self.teacher.parameters()):
    #             teacher_p.data = self.tau * teacher_p.data + (1.0 - self.tau) * p.data

    #         # update tau
    #         max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
    #         self.tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * self.global_step / max_steps) + 1) / 2

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def compute_ood_scores(ensemble_probas: torch.Tensor) -> torch.Tensor:
    mean_entropies = entropy(ensemble_probas.mean(dim=0), dim=-1)
    expected_entropies = entropy(ensemble_probas, dim=-1).mean(dim=0)
    bald_scores = mean_entropies - expected_entropies
    return mean_entropies, expected_entropies, bald_scores


class SensembleOODDetection(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.ood_labels = []
        self.ood_scores = collections.defaultdict(list)

    def on_validation_batch_end(self, trainer, pl_module: Sensemble, outputs, batch, batch_idx, dataloader_idx=0):
        (images, *views), labels = batch
        ood_labels = labels.cpu() == -1

        ood_scores = {}
        with eval_mode(pl_module.encoder):
            ood_scores['entropy'] = entropy(pl_module(images).cpu(), dim=-1)

        with eval_mode(pl_module.encoder, enable_dropout=True):
            ensemble_probas = torch.stack([pl_module(images).cpu() for _ in range(len(views))])
            (ood_scores['mean_entropy'],
             ood_scores['expected_entropy'],
             ood_scores['bald_score']) = compute_ood_scores(ensemble_probas)

            ensemble_probas = torch.stack([pl_module(v).cpu() for v in views])
            (ood_scores['mean_entropy_on_views'],
             ood_scores['expected_entropy_on_views'],
             ood_scores['bald_score_on_views']) = compute_ood_scores(ensemble_probas)

        self.ood_labels.extend(ood_labels.tolist())
        for k, v in ood_scores.items():
            self.ood_scores[k].extend(v.tolist())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for k, v in self.ood_scores.items():
            self.log(f'val/ood_auroc_{k}', roc_auc_score(self.ood_labels, v))
