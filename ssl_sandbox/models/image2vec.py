from typing import *
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pl_bolts.models.autoencoders.components import (
    resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder
)
from pl_bolts.models.self_supervised


class Image2Vec(pl.LightningModule):
    def __init__(
            self,
            image_size: int,
            ae_latent_dim: int,
            ae: bool = True,
            vae: bool = True,
            simclr: bool = True,
            vicreg: bool = True,
            simsiam: bool = True,
            simclr_embed_dim: int = 128,
            vicreg_embed_dim: int = 8192,
            architecture: str = 'resnet50',
            first_conv: bool = False,
            maxpool1: bool = False,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()

        if architecture == 'resnet50':
            encoder_factory = resnet50_encoder
            embed_dim = 2048
            decoder_factory = resnet50_decoder
        elif architecture == 'resnet18':
            encoder_factory = resnet18_encoder
            embed_dim = 512
            decoder_factory = resnet18_decoder
        else:
            raise ValueError(f'architecture {architecture} is not supported.')

        # encoder
        self.encoder = encoder_factory(first_conv, maxpool1)

        # ae/vae stuff
        if ae:
            if vae:
                self.vae_mlp = MLP(embed_dim, embed_dim, 2 * ae_latent_dim)
            else:
                self.ae_mlp = MLP(embed_dim, embed_dim, ae_latent_dim)

            self.decoder = decoder_factory(ae_latent_dim, image_size, first_conv, maxpool1)

        # ssl stuff
        if simclr:
            self.simclr_mlp = MLP(embed_dim, embed_dim, simclr_embed_dim)

        if vicreg:
            self.vicreg_mlp = MLP(embed_dim, embed_dim, vicreg_embed_dim)

        if simsiam:
            self.simsiam_mlp = MLP(embed_dim, embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def compute_ae_loss(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def compute_vae_loss(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def compute_simclr_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass

    def compute_vicreg_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass

    def compute_simsiam_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        (x, x1, x2), _ = batch

        loss = torch.tensor(0.0, requires_grad=True).to(x)

        if self.hparams.ae:
            if self.hparams.vae:
                vae_loss = self.compute_vae_loss(x)
                self.log('losses/vae', vae_loss, on_epoch=True)
                loss += vae_loss
            else:
                ae_loss = self.compute_ae_loss(x)
                self.log('losses/ae', ae_loss, on_epoch=True)
                loss += ae_loss

        if self.hparams.simclr:
            simclr_loss = self.compute_simclr_loss(x1, x2)
            self.log('losses/simclr', simclr_loss, on_epoch=True)
            loss += simclr_loss

        if self.hparams.vicreg:
            vicreg_loss = self.compute_vicreg_loss(x1, x2)
            self.log('losses/vicreg', vicreg_loss, on_epoch=True)
            loss += vicreg_loss

        if self.hparams.simsiam:
            simsiam_loss = self.compute_simsiam_loss(x1, x2)
            self.log('losses/simsiam', simsiam_loss, on_epoch=True)
            loss += simsiam_loss

        self.log('losses/total', loss, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> :
        x, _ = batch
        return self(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
