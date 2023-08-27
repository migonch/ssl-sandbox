from argparse import ArgumentParser

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pl_bolts.datamodules import CIFAR10DataModule

from ssl_sandbox.pretrain import (
    SimCLR, BarlowTwins, VICReg, IBFCodes
)
from ssl_sandbox.eval import OnlineProbing
from ssl_sandbox.pretrain.transforms import SimCLRViews


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cifar10_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--method', required=True)

    parser.add_argument('--dataset', default='cifar10')

    parser.add_argument('--encoder', default='resnet50_cifar10')
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--drop_channel_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--drop_block_rate', type=float, default=0.0)

    parser.add_argument('--barlow_twins_proj_dim', type=int, default=8192)
    parser.add_argument('--vicreg_proj_dim', type=int, default=8192)
    parser.add_argument('--ibf_code_dim', type=int, default=2048)
    parser.add_argument('--ibf_reg_weight', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--clip_grad', type=float)

    parser.add_argument('--ckpt_path')

    return parser.parse_args()


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

    lr = args.base_lr * args.batch_size * torch.cuda.device_count() / 256
    hparams = dict(
        encoder_architecture=args.encoder,
        dropout_rate=args.dropout_rate,
        drop_channel_rate=args.drop_channel_rate,
        drop_block_rate=args.drop_block_rate,
        drop_path_rate=args.drop_path_rate,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        # hparams to save
        batch_size=args.batch_size,
        method=args.method,
        clip_grad=args.clip_grad,
        base_lr=args.base_lr
    )
    match args.method:
        case 'simclr':
            model = SimCLR(
                **hparams
            )
        case 'barlow_twins':
            model = BarlowTwins(
                **hparams,
                proj_dim=args.barlow_twins_proj_dim,
            )
        case 'vicreg':
            model = VICReg(
                **hparams,
                proj_dim=args.vicreg_proj_dim,
            )
        case 'ibf_codes':
            model = IBFCodes(
                **hparams,
                code_dim=args.ibf_code_dim,
                reg_weight=args.ibf_reg_weight
            )
        case _:
            raise ValueError(args.method)

    callbacks = [
        OnlineProbing(model.embed_dim, dm.num_classes),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
    ]

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
