from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP
from ssl_sandbox.nn.functional import off_diagonal


class VICVAE(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            vae_dim: int = 32,
            proj_dim: int = 8192,
            c_weight: float = 0.04,
            kl_weight: float = 1e-6,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 10,
            **hparams: Any
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, enc_dim = encoder(encoder_architecture)
        self.embed_dim = vae_dim
        self.mean_fc = nn.Linear(enc_dim, vae_dim)
        self.logvar_fc = nn.Linear(enc_dim, vae_dim)
        self.projector = nn.Sequential(
            MLP(
                input_dim=vae_dim,
                hidden_dim=proj_dim,
                output_dim=proj_dim,
                num_hidden_layers=2,
                bias=False
            ),
            nn.BatchNorm1d(proj_dim, affine=False)
        )
        self.c_weight = c_weight
        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def forward(self, images):
        return self.mean_fc(self.encoder(images))
    
    def _training_forward(self, x):
        x = self.encoder(x)
        m = self.mean_fc(x)
        s = torch.exp(self.logvar_fc(x) / 2.0) + 1e-5
        q = D.Normal(m, s)
        z = q.rsample()
        z = self.projector(z)
        return z, q

    def training_step(self, batch, batch_idx):
        (_, x_1, x_2), _ = batch

        z_1, q_1 = self._training_forward(x_1)
        z_2, q_2 = self._training_forward(x_2)
        p = D.Normal(torch.zeros_like(q_1.loc), torch.ones_like(q_1.stddev))

        i_reg = F.mse_loss(z_1, z_2)
        self.log(f'pretrain/i_reg', i_reg, on_epoch=True)

        z_1 = z_1 - z_1.mean(dim=0)
        z_2 = z_2 - z_2.mean(dim=0)

        n, d = z_1.shape
        c_reg_1 = off_diagonal(z_1.T @ z_1).div(n - 1).pow_(2).sum().div(d)
        c_reg_2 = off_diagonal(z_2.T @ z_2).div(n - 1).pow_(2).sum().div(d)
        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'pretrain/c_reg', c_reg, on_epoch=True)

        kl_reg_1 = D.kl_divergence(q_1, p).mean()
        kl_reg_2 = D.kl_divergence(q_2, p).mean()
        kl_reg = (kl_reg_1 + kl_reg_2) / 2.0
        self.log('pretrain/kl_reg', kl_reg, on_epoch=True)

        vic_reg = (
            i_reg
            + self.c_weight * c_reg
            + self.kl_weight * kl_reg
        )
        self.log(f'pretrain/vic_reg', vic_reg, on_epoch=True)

        return vic_reg

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
