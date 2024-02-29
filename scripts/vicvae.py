from argparse import ArgumentParser

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pl_bolts.datamodules import CIFAR10DataModule

from ssl_sandbox.pretrain.vicvae import VICVAE
from ssl_sandbox.eval import OnlineProbing
from ssl_sandbox.pretrain.transforms import SimCLRViews


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cifar10_dir', required=True)
    parser.add_argument('--log_dir', required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--clip_grad', type=float)

    parser.add_argument('--ckpt_path')

    return parser.parse_args()


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.cifar10_dir,
        val_split=1000,
        num_workers=args.num_workers,
        normalize=True,
        batch_size=args.batch_size
    )
    image_size = 32
    blur = False
    jitter_strength = 0.5
    dm.train_transforms = SimCLRViews(
        size=image_size,
        jitter_strength=jitter_strength,
        blur=blur,
        final_transforms=dm.default_transforms()
    )
    dm.val_transforms = SimCLRViews(
        size=image_size,
        jitter_strength=jitter_strength,
        blur=blur,
        final_transforms=dm.default_transforms(),
        views_number=10
    )

    lr = args.base_lr * args.batch_size * torch.cuda.device_count() / 256
    hparams = dict(
        encoder_architecture='resnet50_cifar10',
        vae_dim=32,
        proj_dim=8192,
        c_weight=0.04,
        kl_weight=1e-6,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        # hparams to save
        batch_size=args.batch_size,
        clip_grad=args.clip_grad,
        base_lr=args.base_lr
    )
    model = VICVAE(
        **hparams
    )
    callbacks = [
        OnlineProbing(model.embed_dim, dm.num_classes),
        LearningRateMonitor(),
        # DeviceStatsMonitor(),
    ]

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'pretrain/cifar10/vicvae'
    )
    logger.log_hyperparams(model.hparams)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler='simple',
        accelerator='gpu',
        devices=-1,
        # strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=args.num_epochs,
        gradient_clip_val=args.clip_grad,
        log_every_n_steps=10
    )
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main(parse_args())
