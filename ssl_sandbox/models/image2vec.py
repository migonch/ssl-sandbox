from typing import *
import json
import math
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pl_bolts.models.autoencoders.components import (
    resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder
)
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.functional import entropy, off_diagonal
from .blocks import MLP


class Image2Vec(pl.LightningModule):
    def __init__(
            self,
            image_size: int,
            num_classes: int,
            supervised: bool = True,
            ssl_method: Literal['ae', 'vae', 'simclr', 'vicreg', 'qq', 'none'] = 'none',
            ae_dim: int = 128,
            vae_dim: int = 128,
            vae_beta: float = 0.1,
            simclr_dim: int = 128,
            simclr_temp: float = 0.1,
            simclr_decoupled: bool = False,
            vicreg_dim: int = 2048,
            i_weight: float = 25.0,
            v_weight: float = 25.0,
            qq_num_predicates: int = 2048,
            qq_sharpen_temp: float = 0.25,
            qq_reg_weight: float = 1.0,
            architecture: Literal['resnet18', 'resnet50'] = 'resnet50',
            first_conv: bool = True,
            maxpool1: bool = True,
            lr: float = 1e-2,
            weight_decay: float = 1e-4,
            warmup_epochs: int = 100
    ) -> None:
        assert ssl_method in ['ae', 'vae', 'simclr', 'vicreg', 'qq', 'none']
        assert supervised or ssl_method != 'none'

        super().__init__()

        self.save_hyperparameters()

        if architecture == 'resnet18':
            encoder_factory = resnet18_encoder
            vec_dim = 512
            decoder_factory = resnet18_decoder
        elif architecture == 'resnet50':
            encoder_factory = resnet50_encoder
            vec_dim = 2048
            decoder_factory = resnet50_decoder
        else:
            raise ValueError(f'architecture {architecture} is not supported.')

        # encoder
        self.encoder = encoder_factory(first_conv, maxpool1)

        # classification head
        self.cls_head = nn.Linear(vec_dim, num_classes)

        # ae or vae stuff
        if ssl_method == 'ae':
            assert ae_dim is not None
            self.ae_head = MLP(vec_dim, vec_dim, ae_dim)
            self.decoder = decoder_factory(ae_dim, image_size, first_conv, maxpool1)
        if ssl_method == 'vae':
            assert vae_dim is not None
            self.vae_head = MLP(vec_dim, vec_dim, 2 * vae_dim)
            self.decoder = decoder_factory(vae_dim, image_size, first_conv, maxpool1)
        if ssl_method == 'simclr':
            self.simclr_head = MLP(vec_dim, vec_dim, simclr_dim)
        if ssl_method == 'vicreg':
            self.vicreg_head = MLP(vec_dim, vicreg_dim, vicreg_dim)
        if ssl_method == 'qq':
            self.qq_head = MLP(vec_dim, qq_num_predicates, qq_num_predicates)
            # self.qq_teacher = deepcopy(nn.Sequential(self.encoder, self.qq_head))

        self.automatic_optimization = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def cls_step(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.hparams.supervised:
            with torch.no_grad():
                vecs = self(images)
        else:
            vecs = self(images)
        loss = F.cross_entropy(self.cls_head(vecs), labels)
        self.log('train/cls_loss', loss, on_epoch=True)
        self.manual_backward(loss)

    def ae_step(self, images: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(self.decoder(self.ae_head(self(images))), images)
        self.log('train/ae_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def vae_step(self, images: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.split(self.vae_head(self(images)), self.hparams.vae_dim, dim=1)
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mean, std)
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))

        recon_loss = F.mse_loss(self.decoder(q.rsample()), images)
        kl = torch.distributions.kl_divergence(q, p).mean()

        loss = recon_loss + self.hparams.vae_beta * kl
        self.log('train/vae_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def simclr_step(self, images_1: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        embeds_1 = F.normalize(self.simclr_head(self(images_1)), dim=1)  # (batch_size, simclr_dim)
        embeds_2 = F.normalize(self.simclr_head(self(images_2)), dim=1)  # (batch_size, simclr_dim)

        temp = self.hparams.simclr_temp
        logits_11 = torch.matmul(embeds_1, embeds_1.T) / temp
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(embeds_1, embeds_2.T) / temp
        pos_logits = logits_12.diag()
        if self.hparams.simclr_decoupled:
            logits_12.fill_diagonal_(float('-inf'))
        logits_22 = torch.matmul(embeds_2, embeds_2.T) / temp
        logits_22.fill_diagonal_(float('-inf'))

        loss_1 = torch.mean(-pos_logits + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        loss_2 = torch.mean(-pos_logits + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        loss = (loss_1 + loss_2) / 2
        self.log(f'train/simclr_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def vicreg_step(self, images_1: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        embeds_1 = self.vicreg_head(self(images_1))
        embeds_2 = self.vicreg_head(self(images_2))

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'train/i_reg', i_reg, on_epoch=True)

        embeds_1 = embeds_1 - embeds_1.mean(dim=0)
        embeds_2 = embeds_2 - embeds_2.mean(dim=0)

        eps = 1e-4
        v_reg_1 = torch.mean(F.relu(1 - torch.sqrt(embeds_1.var(dim=0) + eps)))
        v_reg_2 = torch.mean(F.relu(1 - torch.sqrt(embeds_2.var(dim=0) + eps)))
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log(f'train/v_reg', v_reg, on_epoch=True)

        n, d = embeds_1.shape
        c_reg_1 = off_diagonal(embeds_1.T @ embeds_1).div(n - 1).pow_(2).sum().div(d)
        c_reg_2 = off_diagonal(embeds_2.T @ embeds_2).div(n - 1).pow_(2).sum().div(d)
        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'train/c_reg', c_reg, on_epoch=True)

        vic_reg = self.hparams.i_weight * i_reg + self.hparams.v_weight * v_reg + c_reg
        self.log(f'train/vic_reg', vic_reg, on_epoch=True)

        self.manual_backward(vic_reg, retain_graph=True)

    @staticmethod
    def qq_reg(logits):
        p = torch.sigmoid(logits)  # (N, K)
        p = torch.stack([1 - p, p])  # (2, N, K)
        priors = p.transpose(-2, -1) @ p.unsqueeze(1) / p.shape[1]  # (2, 2, K, K)
        return 2 * math.log(2) - off_diagonal(entropy(priors, dim=(0, 1))).mean()

    def qq_step(self, images_1: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        logits_1 = self.qq_head(self(images_1))
        logits_2 = self.qq_head(self(images_2))

        temp = self.hparams.qq_sharpen_temp
        bootstrap_loss_1 = F.binary_cross_entropy_with_logits(logits_1, torch.sigmoid(logits_2.detach() / temp))
        bootstrap_loss_2 = F.binary_cross_entropy_with_logits(logits_2, torch.sigmoid(logits_1.detach() / temp))
        bootstrap_loss = (bootstrap_loss_1 + bootstrap_loss_2) / 2
        self.log(f'train/qq_bootstrap_loss', bootstrap_loss, on_epoch=True)

        reg = (self.qq_reg(logits_1) + self.qq_reg(logits_2)) / 2
        self.log(f'train/qq_reg', reg, on_epoch=True)

        loss = bootstrap_loss + self.hparams.qq_reg_weight * reg
        self.log(f'train/qq_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        (images, images_1, images_2), labels = batch

        optimizer = self.optimizers()
        optimizer.zero_grad()

        if self.hparams.ssl_method == 'ae':
            self.ae_step(images)
        if self.hparams.ssl_method == 'vae':
            self.vae_step(images)
        if self.hparams.ssl_method == 'simclr':
            self.simclr_step(images_1, images_2)
        if self.hparams.ssl_method == 'vicreg':
            self.vicreg_step(images_1, images_2)
        if self.hparams.ssl_method == 'qq':
            self.qq_step(images_1, images_2)
        self.cls_step(images, labels)

        # can help to stabilize the training
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()

        if self.trainer.is_last_batch:
            scheduler = self.lr_schedulers()
            scheduler.step()

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        vecs = self(images)
        acc = accuracy(
            preds=self.cls_head(vecs),
            target=labels,
            task='multiclass',
            num_classes=self.hparams.num_classes
        )
        self.log('val/accuracy', acc)

        embeds = {'vecs': vecs}
        if self.hparams.ssl_method == 'ae':
            embeds['ae_embeddings'] = self.ae_head(vecs)
        if self.hparams.ssl_method == 'vae':
            embeds['vae_embeddings'] = self.vae_head(vecs)[:, :self.hparams.vae_dim]
        if self.hparams.ssl_method == 'simclr':
            embeds['simclr_embeddings'] = F.normalize(self.simclr_head(vecs), dim=1)
        if self.hparams.ssl_method == 'vicreg':
            embeds['vicreg_embeddings'] = self.vicreg_head(vecs)
        if self.hparams.ssl_method == 'qq':
            embeds['qq_logits'] = self.qq_head(vecs)
        return embeds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class QQTeacherUpdate(pl.Callback):
    def __init__(self, initial_tau: float = 0.996) -> None:
        if not 0.0 <= initial_tau <= 1.0:
            raise ValueError(f"initial tau should be between 0 and 1 instead of {initial_tau}.")

        super().__init__()

        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(self, trainer, pl_module: Image2Vec, outputs, batch, batch_idx) -> None:
        student = nn.Sequential(pl_module.encoder, pl_module.qq_head)
        teacher = pl_module.qq_teacher
        self.update_weights(student, teacher)

        self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module, trainer) -> None:
        """Update tau value for next update."""
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        self.current_tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2

    def update_weights(self, student, teacher) -> None:
        """Update target network parameters."""
        for student_w, teacher_w in zip(student.parameters(), teacher.parameters()):
            teacher_w.data = self.current_tau * teacher_w.data + (1 - self.current_tau) * student_w.data


class LogEmbeddings(pl.Callback):
    def __init__(self):
        super().__init__()

        self.data = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, labels = batch
        self.data.extend([
            {k: v[i].tolist() for k, v in outputs.items()} | {'label': str(l.item())}
            for i, l in enumerate(labels)
        ])

    def on_validation_epoch_end(self, trainer, pl_module):
        logger: TensorBoardLogger = trainer.logger

        with open(f'{logger.log_dir}/embeddings.json', 'w') as f:
            json.dump(self.data, f)

        # clear data after each epoch
        self.data = []
