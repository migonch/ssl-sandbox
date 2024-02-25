from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pl_bolts.datamodules import CIFAR10DataModule

from ssl_sandbox.pretrain.wasreg import WassersteinReg
from ssl_sandbox.eval import OnlineProbing
from ssl_sandbox.pretrain.transforms import SimCLRViews


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cifar10_dir', required=True)
    parser.add_argument('--log_dir', required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--num_epochs', type=int, default=5000)

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

    model_kwargs = dict(
        encoder_architecture='resnet50_cifar10',
        projector_hidden_dim=2048,
        projector_out_dim=512,
        critic_hidden_dim=1024,
        gp_weight=100.0,
        num_critic_steps=5,
        wasreg_weight=0.01,
        lr=3e-4,
        betas=(0.5, 0.9),
        weight_decay=1e-6,
        # hparams to save
        batch_size=args.batch_size,
    )
    model = WassersteinReg(**model_kwargs)

    callbacks = [
        OnlineProbing(model.embed_dim, dm.num_classes),
        LearningRateMonitor(),
        # DeviceStatsMonitor(),
    ]

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'pretrain/cifar10/wasreg'
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
        log_every_n_steps=10
    )
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main(parse_args())
