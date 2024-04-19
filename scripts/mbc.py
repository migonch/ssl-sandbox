from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pl_bolts.datamodules import CIFAR10DataModule

from ssl_sandbox.pretrain.mbc import MutualBinaryCodes
from ssl_sandbox.eval import OnlineProbing
from ssl_sandbox.pretrain.transforms import SimCLRViews


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cifar10_dir', required=True)
    parser.add_argument('--log_dir', required=True)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--ckpt_path')

    return parser.parse_args()


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.cifar10_dir,
        val_split=1000,
        num_workers=args.num_workers,
        normalize=True,
        batch_size=args.batch_size,
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

    model = MutualBinaryCodes(
        encoder_architecture='resnet50_cifar10',
        num_heads=1024,
        head_hidden_dim=2048,
        lr=1e-2,
        weight_decay=1e-6,
        epochs=args.epochs,
        warmup_epochs=10,
        batches_per_epoch=dm.num_samples // args.batch_size
    )

    callbacks = [
        OnlineProbing(model.repr_dim, dm.num_classes),
        LearningRateMonitor(),
        # DeviceStatsMonitor(),
    ]

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'pretrain/cifar10/mbc'
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler='simple',
        accelerator='gpu',
        devices=-1,
        # strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=args.epochs,
        precision=16
    )
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main(parse_args())
