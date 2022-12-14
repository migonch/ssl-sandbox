from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models import VAE


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cifar_dir")
    parser.add_argument("--logs_dir")
    return parser.parse_args()


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.cifar_dir,
        num_workers=8,
        normalize=True,
        batch_size=256
    )
    model = VAE(
        input_height=32,
        latent_dim=128,
    )
    logger = WandbLogger(
        save_dir=args.logs_dir,
        project='ssl-sandbox'
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=1,
        max_epochs=100
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)
