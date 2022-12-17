from typing import *

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pl_bolts.models.autoencoders.components import (
    resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder
)

from .blocks import MLP


class Image2Vec(pl.LightningModule):
    def __init__(
            self,
            image_size: int,
            num_classes: int,
            supervised: bool = True,
            # ae/vae stuff
            ae: bool = False,
            ae_latent_dim: Optional[int] = None,
            vae: bool = False,
            vae_latent_dim: Optional[int] = None,
            vae_beta: float = 0.1,
            # ssl stuff
            simclr: bool = False,
            simclr_embed_dim: int = 128,
            simclr_temperature: float = 0.1,
            vicreg: bool = False,
            vicreg_embed_dim: int = 2048,
            simsiam: bool = False,
            # architecture
            architecture: str = 'resnet18',
            first_conv: bool = False,
            maxpool1: bool = False,
            # optimization
            lr: float = 3e-4,
    ) -> None:
        assert supervised or ae or vae or simclr or vicreg or simsiam
        assert not (ae and vae)

        super().__init__()

        self.save_hyperparameters()

        if architecture == 'resnet18':
            encoder_factory = resnet18_encoder
            feat_dim = 512
            decoder_factory = resnet18_decoder
        elif architecture == 'resnet50':
            encoder_factory = resnet50_encoder
            feat_dim = 2048
            decoder_factory = resnet50_decoder
        else:
            raise ValueError(f'architecture {architecture} is not supported.')

        # encoder
        self.encoder = encoder_factory(first_conv, maxpool1)

        # classification head
        self.cls_mlp = MLP(feat_dim, feat_dim, num_classes)

        # ae or vae stuff
        if ae:
            assert ae_latent_dim is not None
            self.ae_mlp = MLP(feat_dim, feat_dim, ae_latent_dim)
            self.decoder = decoder_factory(ae_latent_dim, image_size, first_conv, maxpool1)
        elif vae:
            assert vae_latent_dim is not None
            self.vae_mlp = MLP(feat_dim, feat_dim, 2 * vae_latent_dim)
            self.decoder = decoder_factory(vae_latent_dim, image_size, first_conv, maxpool1)

        # ssl stuff
        if simclr:
            self.simclr_mlp = MLP(feat_dim, feat_dim, simclr_embed_dim)

        if vicreg:
            self.vicreg_mlp = MLP(feat_dim, feat_dim, vicreg_embed_dim)

        if simsiam:
            self.simsiam_mlp = MLP(feat_dim, feat_dim, feat_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def compute_cls_loss(self, features: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        if not self.hparams.supervised:
            features = features.detach()
        return F.cross_entropy(self.cls_mlp(features), gt_labels)

    def compute_ae_loss(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(self.decoder(self.ae_mlp(features)), images)
        logs = {'ae_recon_loss': loss}
        return loss, logs

    def compute_vae_loss(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.split(self.vae_mlp(features), self.hparams.ae_latent_dim, dim=1)
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mean, std)
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))

        recon_loss = F.mse_loss(self.decoder(q.rsample()), images)
        kl = torch.distributions.kl_divergence(q, p).mean()

        loss = recon_loss + self.hparams.vae_beta * kl
        logs = {
            'vae_recon_loss': recon_loss,
            'vae_kl': kl,
            'vae_loss': loss,
        }
        return loss, logs

    def compute_simclr_loss(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        embeddings_1 = F.normalize(self.simclr_mlp(features_1), dim=1)  # (batch_size, simclr_embed_dim)
        embeddings_2 = F.normalize(self.simclr_mlp(features_2), dim=1)  # (batch_size, simclr_embed_dim)
        sim = torch.matmul(embeddings_1, embeddings_2.T)  # (batch_size, batch_size)
        return (sim / self.hparams.simclr_temperature).log_softmax(dim=1).diag().mean()

    def compute_vicreg_loss(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_simsiam_loss(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        (images, images_1, images_2), gt_labels = batch
        features, features_1, features_2 = self(images), self(images_1), self(images_2)

        loss = self.compute_cls_loss(features, gt_labels)
        self.log('train/cls_loss', loss, on_epoch=True)

        if self.hparams.ae:
            ae_loss, ae_logs = self.compute_ae_loss(features, images)
            self.log_dict({f"train/{k}": v for k, v in ae_logs.items()}, on_step=True, on_epoch=True)
            loss += ae_loss

        if self.hparams.vae:
            vae_loss, vae_logs = self.compute_vae_loss(features, images)
            self.log_dict({f"train/{k}": v for k, v in vae_logs.items()}, on_step=True, on_epoch=True)
            loss += vae_loss

        if self.hparams.simclr:
            simclr_loss = self.compute_simclr_loss(features_1, features_2)
            self.log('train/simclr_loss', simclr_loss, on_epoch=True)
            loss += simclr_loss

        if self.hparams.vicreg:
            vicreg_loss = self.compute_vicreg_loss(features_1, features_2)
            self.log('train/vicreg_loss', vicreg_loss, on_epoch=True)
            loss += vicreg_loss

        if self.hparams.simsiam:
            simsiam_loss = self.compute_simsiam_loss(features_1, features_2)
            self.log('train/simsiam_loss', simsiam_loss, on_epoch=True)
            loss += simsiam_loss

        self.log('train/total_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, gt_labels = batch
        features = self(images)
        pred_labels = self.cls_mlp(features).argmax(dim=1)
        self.log('val/accuracy', accuracy(pred_labels, gt_labels))
        return features

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class TrainDataTransform:
    """Default augmentations described in https://arxiv.org/pdf/2002.05709.pdf.
    Code is similar to pl_bolts.models.self_supervised.simclr.transforms.SimCLRTrainDataTransform.
    """

    def __init__(
            self,
            image_size: int,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.0,
            normalize: Optional[nn.Module] = None
    ) -> None:
        """Initialize transforms.

        Args:
            image_size (int, optional): input image size.
            gaussian_blur (bool, optional): Whether to apply gaussian blur. Defaults to True.
            jitter_strengh (float, optional): Strength of color distortion. Defaults to 1.0.
            normalize (nn.Module, optional): normalization.
        """
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * jitter_strength,
            contrast=0.8 * jitter_strength,
            saturation=0.8 * jitter_strength,
            hue=0.2 * jitter_strength
        )

        augmentations = [
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if gaussian_blur:
            kernel_size = int(0.1 * image_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            augmentations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size)], p=0.5))

        self.augmentations = transforms.Compose(augmentations)
        self.final_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize or nn.Identity()
        ])

    def __call__(self, images):
        images_1 = self.final_transforms(self.augmentations(images))
        images_2 = self.final_transforms(self.augmentations(images))
        images = self.final_transforms(images)
        return images, images_1, images_2


class LogEmbeddings(pl.Callback):
    def __init__(self):
        super().__init__()

        self.data = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, gt_labels = batch
        self.data.extend([[e.tolist(), str(l.item())] for e, l in zip(outputs, gt_labels)])

    def on_validation_epoch_end(self, trainer, pl_module):
        logger: WandbLogger = trainer.logger

        logger.log_table(
            key='embeddings',
            columns=['embedding', 'label'],
            data=self.data
        )

        # clear data after each epoch
        self.data = []
