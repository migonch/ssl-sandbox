from typing import *
import json
import random
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pl_bolts.models.autoencoders.components import (
    resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder
)

from ssl_sandbox.functional import entropy, off_diagonal
from .blocks import MLP


class Image2Vec(pl.LightningModule):
    def __init__(
            self,
            image_size: int,
            num_classes: int,
            supervised: bool = True,
            ae: bool = False,
            ae_dim: Optional[int] = None,
            vae: bool = False,
            vae_dim: Optional[int] = None,
            vae_beta: float = 0.1,
            simclr: bool = False,
            simclr_dim: int = 128,
            simclr_temp: float = 0.1,
            vicreg: bool = False,
            vicreg_dim: int = 8192,
            i_weight: float = 25.0,
            v_weight: float = 25.0,
            qq: bool = False,
            qq_num_predicates: int = 8192,
            qq_sharpen_temp: float = 0.25,
            qq_reg_weight: float = 1e2,
            architecture: Literal['resnet18', 'resnet50'] = 'resnet18',
            first_conv: bool = True,
            maxpool1: bool = True,
            lr: float = 3e-4,
    ) -> None:
        assert supervised or ae or vae or simclr or vicreg
        assert not (ae and vae)

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
        if ae:
            assert ae_dim is not None
            self.ae_mlp = MLP(vec_dim, vec_dim, ae_dim)
            self.decoder = decoder_factory(ae_dim, image_size, first_conv, maxpool1)
        elif vae:
            assert vae_dim is not None
            self.vae_mlp = MLP(vec_dim, vec_dim, 2 * vae_dim)
            self.decoder = decoder_factory(vae_dim, image_size, first_conv, maxpool1)

        # ssl stuff
        if simclr:
            self.simclr_mlp = MLP(vec_dim, vec_dim, simclr_dim)

        if vicreg:
            self.vicreg_mlp = MLP(vec_dim, vicreg_dim, vicreg_dim)

        if qq:
            self.qq_head = MLP(vec_dim, qq_num_predicates, qq_num_predicates)
            self.qq_priors = torch.full((qq_num_predicates, qq_num_predicates, 4), 1 / 4)  # only upper triangle is used

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
        loss = F.mse_loss(self.decoder(self.ae_mlp(self(images))), images)
        self.log('train/ae_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def vae_step(self, images: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.split(self.vae_mlp(self(images)), self.hparams.vae_dim, dim=1)
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mean, std)
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))

        recon_loss = F.mse_loss(self.decoder(q.rsample()), images)
        kl = torch.distributions.kl_divergence(q, p).mean()

        loss = recon_loss + self.hparams.vae_beta * kl
        self.log('train/vae_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def simclr_step(self, images_1: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        embeds_1 = F.normalize(self.simclr_mlp(self(images_1)), dim=1)  # (batch_size, simclr_dim)
        embeds_2 = F.normalize(self.simclr_mlp(self(images_2)), dim=1)  # (batch_size, simclr_dim)

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
        embeds_1 = self.vicreg_mlp(self(images_1))
        embeds_2 = self.vicreg_mlp(self(images_2))
        
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

    def qq_step(self, images_1: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        N = self.hparams.qq_num_predicates
        n = self.hparams.qq_num_predicates_per_iter
        indices = random.sample(range(N), n)
        qq_weight = self.qq_head.weight[indices]
        qq_bias = self.qq_head.bias[indices]

        logits = F.linear(self(images_1), qq_weight, qq_bias)

        with torch.no_grad():
            targets = torch.sigmoid(F.linear(self(images_2), qq_weight, qq_bias) / self.hparams.qq_sharpen_temp)
        bootstrap_loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.log(f'train/qq_bootstrap_loss', bootstrap_loss, on_epoch=True)

        p = torch.sigmoid(logits)  # (b, n)
        p = torch.stack([1 - p, p], dim=-1)  # (b, n, 2)
        first, second = torch.triu_indices(n, n)
        batch_priors = torch.mean(p[:, first, :, None] * p[:, second, None], dim=0).flatten(-2)  # (b, n(n-1)/2, 4)
        historical_priors = self.pairwise_joint_priors[indices[first], indices[second]].to(batch_priors)
        priors = (1 - self.gamma) * batch_priors + self.gamma * historical_priors
        self.qq_priors[indices[first], indices[second]] = priors.data.cpu()  # update priors
        reg = 2 * math.log(2) - entropy(priors, dim=-1).mean()
        self.log(f'train/qq_reg', reg, on_epoch=True)

        loss = bootstrap_loss + self.reg_weight * reg
        self.log(f'train/qq_loss', loss, on_epoch=True)
        self.manual_backward(loss, retain_graph=True)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        (images, images_1, images_2), labels = batch

        optimizer = self.optimizers()
        optimizer.zero_grad()

        if self.hparams.ae:
            self.ae_step(images)
        if self.hparams.vae:
            self.vae_step(images)
        if self.hparams.simclr:
            self.simclr_step(images_1, images_2)
        if self.hparams.vicreg:
            self.vicreg_step(images_1, images_2)
        if self.hparams.qq:
            self.qq_step(images_1, images_2)
        self.cls_step(images, labels)

        # can help to stabilize the training
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()

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
        if self.hparams.ae:
            embeds['ae_embeddings'] = self.ae_mlp(vecs)
        if self.hparams.vae:
            embeds['vae_embeddings'] = self.vae_mlp(vecs)[:, :self.hparams.vae_dim]
        if self.hparams.simclr:
            embeds['simclr_embeddings'] = F.normalize(self.simclr_mlp(vecs), dim=1)
        if self.hparams.vicreg:
            embeds['vicreg_embeddings'] = self.vicreg_mlp(vecs)
        return embeds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


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
