from argparse import ArgumentParser

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pl_bolts.datamodules import CIFAR10DataModule

import timm

from ssl_sandbox.nn.resnet import resnet50
from ssl_sandbox.pretrain import (
    SimCLR, BarlowTwins, BarlowTwinsOODDetection, VICReg, VICRegOODDetection, Sensemble
)
from ssl_sandbox.eval import OnlineProbing
from ssl_sandbox.datamodules import CIFAR4vs6DataModule
from ssl_sandbox.pretrain.transforms import SimCLRViews

 
def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cifar10_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--method', required=True)

    parser.add_argument('--encoder', default='resnet50')
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--drop_channel_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--drop_block_rate', type=float, default=0.0)

    parser.add_argument('--barlow_twins_unbiased', default=False, action='store_true')
    parser.add_argument('--barlow_twins_proj_dim', type=int, default=8192)
    parser.add_argument('--vicreg_proj_dim', type=int, default=8192)
    parser.add_argument('--vicreg_i_weight', type=float, default=25.0)
    parser.add_argument('--sensemble_num_prototypes', type=int, default=1024)
    parser.add_argument('--sensemble_memax_weight', type=float, default=1.0)
    parser.add_argument('--sensemble_num_sinkhorn_iters', type=int, default=3)
    parser.add_argument('--sensemble_ema', default=False, action='store_true')

    parser.add_argument('--batch_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--clip_grad', type=float)

    parser.add_argument('--ckpt_path')

    return parser.parse_args()


def adapt_to_cifar10(resnet: timm.models.ResNet):
    """See https://arxiv.org/pdf/2002.05709.pdf, Appendix B.9.
    """
    resnet.conv1 = nn.Conv2d(resnet.conv1.in_channels, resnet.conv1.out_channels,
                             kernel_size=3, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    return resnet


def main(args):
    match args.dataset.lower():
        case 'cifar10':
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
        case 'cifar4vs6':
            dm = CIFAR4vs6DataModule(
                data_dir=args.cifar10_dir,
                val_split=1000,
                num_workers=args.num_workers,
                normalize=True,
                batch_size=args.batch_size,
            )
            image_size = 32
            blur = False
            jitter_strength = 0.5
        case _:
            raise ValueError(args.dataset)
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

    dropout_params = dict(
        dropout_rate=args.dropout_rate,
        drop_channel_rate=args.drop_channel_rate,
        drop_block_rate=args.drop_block_rate,
        drop_path_rate=args.drop_path_rate,
    )
    match args.encoder:
        case 'resnet50':
            embed_dim = 2048
            encoder = resnet50(**dropout_params)
            encoder.fc = nn.Identity()
            if args.dataset in ['cifar10', 'cifar4vs6']:
                encoder = adapt_to_cifar10(encoder)
        case _:
            raise ValueError(args.encoder)
    
    lr = args.base_lr * args.batch_size * torch.cuda.device_count() / 256
    optimizer_kwargs = dict(
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs
    )
    hparams = dict(
        batch_size=args.batch_size,
        method=args.method,
        encoder_architecture=args.encoder,
        **dropout_params,
        clip_grad=args.clip_grad,
        base_lr=args.base_lr
    )
    match args.method:
        case 'simclr':
            model = SimCLR(
                encoder,
                embed_dim,
                **optimizer_kwargs,
                **hparams
            )
        case 'barlow_twins':
            model = BarlowTwins(
                encoder,
                embed_dim,
                proj_dim=args.barlow_twins_proj_dim,
                unbiased=args.barlow_twins_unbiased,
                **optimizer_kwargs,
                **hparams
            )
        case 'vicreg':
            model = VICReg(
                encoder,
                embed_dim,
                proj_dim=args.vicreg_proj_dim,
                i_weight=args.vicreg_i_weight,
                **optimizer_kwargs,
                **hparams
            )
        case 'sensemble':
            model = Sensemble(
                encoder,
                embed_dim,
                num_prototypes=args.sensemble_num_prototypes,
                memax_weight=args.sensemble_memax_weight,
                num_sinkhorn_iters=args.sensemble_num_sinkhorn_iters,
                ema=args.sensemble_ema,
                **optimizer_kwargs,
                **hparams
            )
        case _:
            raise ValueError(args.method)

    callbacks = [
        OnlineProbing(embed_dim, dm.num_classes),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
    ]
    if args.method == 'barlow_twins':
        callbacks.append(BarlowTwinsOODDetection())
    if args.method == 'vicreg':
        callbacks.append(VICRegOODDetection())
    if args.method == 'sensemble':
        callbacks.append(
            ModelCheckpoint(save_top_k=1, monitor='val/ood_auroc_mean_entropy_on_views', filename='best', mode='max')
        )

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'pretrain/{args.dataset.lower()}/{args.method}'
    )
    logger.log_hyperparams(model.hparams)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler='simple',
        accelerator='gpu',
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=args.num_epochs,
        gradient_clip_val=args.clip_grad,
        log_every_n_steps=10
    )
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main(parse_args())
