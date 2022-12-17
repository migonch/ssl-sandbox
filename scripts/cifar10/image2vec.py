from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from ssl_sandbox.models.image2vec import Image2Vec, TrainDataTransform, LogEmbeddings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cifar_dir')
    parser.add_argument('--logs_dir')
    parser.add_argument('--name')

    parser.add_argument('--supervised', default=False, action='store_true')
    parser.add_argument('--ae', default=False, action='store_true')
    parser.add_argument('--vae', default=False, action='store_true')
    parser.add_argument('--simclr', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=0)

    return parser.parse_args()


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.cifar_dir,
        num_workers=args.num_workers,
        normalize=True,
        batch_size=args.batch_size,
        val_split=1000,
    )
    dm.train_transforms = TrainDataTransform(
        image_size=dm.dims[1],
        gaussian_blur=False,
        jitter_strength=0.5,
        normalize=cifar10_normalization()
    )
    model = Image2Vec(
        image_size=dm.dims[-1],
        num_classes=dm.num_classes,
        supervised=args.supervised,
        ae=args.ae,
        ae_latent_dim=128,
        vae=args.vae,
        vae_latent_dim=128,
        simclr=args.simclr,
        lr=args.lr
    )
    logger = WandbLogger(
        name=args.name,
        save_dir=args.logs_dir,
        project='ssl-sandbox'
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), LogEmbeddings()],
        accelerator='gpu',
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)
