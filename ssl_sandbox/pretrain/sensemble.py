import collections
from sklearn.metrics import roc_auc_score

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
            sharpen_temp: float = 0.25,
            memax_reg_weight: float = 1.0,
            symmetric: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = Projector(embed_dim, num_prototypes)

        self.temp = temp
        self.sharpen_temp = sharpen_temp
        self.memax_reg_weight = memax_reg_weight
        self.symmetric = symmetric
    
    def forward(self, images):
        return torch.softmax(self.projector(self.encoder(images)) / self.temp)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        logits_1 = self.projector(self.encoder(views_1)) / self.temp  # (batch_size, num_prototypes)
        logits_2 = self.projector(self.encoder(views_2)) / self.temp  # (batch_size, num_prototypes)

        target_1 = torch.softmax(logits_2.detach() / self.sharpen_temp, dim=-1)
        bootstrap_loss = F.cross_entropy(logits_1, target_1)

        probas_1 = torch.softmax(logits_1, dim=-1)
        memax_reg = -entropy(probas_1.mean(dim=0), dim=-1)

        if self.symmetric:
            target_2 = torch.softmax(logits_1.detach() / self.sharpen_temp, dim=-1)
            bootstrap_loss += F.cross_entropy(logits_2, target_2)
            bootstrap_loss /= 2

            probas_2 = torch.softmax(logits_2, dim=-1)
            memax_reg -= entropy(probas_2, dim=-1)
            memax_reg /= 2

        loss = bootstrap_loss + self.memax_reg_weight * memax_reg

        self.log(f'train/bootstrap_loss', bootstrap_loss, on_epoch=True)
        self.log(f'train/memax_reg', memax_reg, on_epoch=True)
        self.log(f'train/loss', loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def compute_ood_scores(ensemble_probas: torch.Tensor) -> torch.Tensor:
    mean_entropies = entropy(probas.mean(dim=0), dim=-1)
    expected_entropies = entropy(probas, dim=-1).mean(dim=0)
    bald_scores = mean_entropies - expected_entropies
    return mean_entropies, expected_entropies, bald_scores


class ValidateOODDetection(pl.Callback):
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
        for k, v in self.odd_scores.items():
            self.log(f'val/ood_auroc_{k}', roc_auc_score(self.ood_labels, v))
