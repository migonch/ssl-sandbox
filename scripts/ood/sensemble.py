from argparse import ArgumentParser

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from ssl_sandbox.pretrain import Sensemble
from ssl_sandbox.eval import OnlineProbing
from ssl_sandbox.datamodules import CIFAR4vs6DataModule
from ssl_sandbox.pretrain.transforms import SimCLRViews, SensembleTrainViews, SensembleInferenceViews


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--cifar10_dir')

    parser.add_argument('--encoder_architecture', default='resnet50_cifar10')
    parser.add_argument('--dropout_rate', type=float, default=0.25)
    parser.add_argument('--drop_channel_rate', type=float, default=0.25)
    parser.add_argument('--drop_block_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    
    parser.add_argument('--views', default='simclr')

    parser.add_argument('--num_prototypes', type=int, default=2048)
    parser.add_argument('--prototype_dim', type=int, default=128)
    parser.add_argument('--sinkhorn_queue_size', type=int, default=3072)
    parser.add_argument('--memax_weight', type=float, default=1.0)
    parser.add_argument('--dispersion_weight', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10_000)
    parser.add_argument('--clip_grad', type=float)

    parser.add_argument('--ckpt_path')

    return parser.parse_args()


def main(args):
    dm = CIFAR4vs6DataModule(
        data_dir=args.cifar10_dir,
        val_split=1000,
        num_workers=args.num_workers,
        normalize=True,
        batch_size=args.batch_size,
    )
    image_size = 32
    jitter_strength = 0.5
    blur = False
    match args.views:
        case 'simclr':
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
        case 'sensemble':
            dm.train_transforms = SensembleTrainViews(
                size=image_size,
                blur=blur,
                final_transforms=dm.default_transforms()
            )
            dm.val_transforms = SensembleInferenceViews(
                size=image_size,
                blur=blur,
                final_transforms=dm.default_transforms(),
                views_number=10
            )
        case _:
            raise ValueError(args.views)

    lr = args.base_lr * args.batch_size * torch.cuda.device_count() / 256
    model = Sensemble(
        encoder_architecture=args.encoder_architecture,
        dropout_rate=args.dropout_rate,
        drop_channel_rate=args.drop_channel_rate,
        drop_block_rate=args.drop_block_rate,
        drop_path_rate=args.drop_path_rate,
        prototype_dim=args.prototype_dim,
        num_prototypes=args.num_prototypes,
        sinkhorn_queue_size=args.sinkhorn_queue_size,
        memax_weight=args.memax_weight,
        dispersion_weight=args.dispersion_weight,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        # hparams to save
        views=args.views,
        batch_size=args.batch_size,
        clip_grad=args.clip_grad,
        base_lr=args.base_lr
    )

    callbacks = [
        OnlineProbing(model.embed_dim, dm.num_classes),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        ModelCheckpoint(save_top_k=1, monitor='val/ood_auroc_mean_entropy_on_views',
                        filename='best', mode='max', save_last=True)
    ]

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'pretrain/cifar4vs6/sensemble'
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
