from argparse import ArgumentParser

from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from ssl_sandbox.models.image2vec import Image2Vec, TrainDataTransform, LogEmbeddings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--mnist_dir')
    parser.add_argument('--cifar10_dir')
    parser.add_argument('--logs_dir')

    parser.add_argument('--supervised', default=False, action='store_true')
    parser.add_argument('--ae', default=False, action='store_true')
    parser.add_argument('--ae_latent_dim', type=int, default=128)
    parser.add_argument('--vae', default=False, action='store_true')
    parser.add_argument('--vae_latent_dim', type=int, default=128)
    parser.add_argument('--simclr', default=False, action='store_true')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoint')

    return parser.parse_args()


def main(args):
    if args.dataset == 'mnist':
        dm = MNISTDataModule(
            data_dir=args.mnist_dir,
            num_workers=args.num_workers,
            normalize=True,
            batch_size=args.batch_size,
            val_split=1000
        )
        mnist_transforms = transforms.Compose([
            dm.default_transforms(),
            transforms.Resize(size=32),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        dm.train_transforms = TrainDataTransform(
            image_size=dm.dims[-1],
            gaussian_blur=False,
            jitter_strength=0.5,
            final_transforms=mnist_transforms
        )
        dm.val_transforms = dm.test_transforms = mnist_transforms
    elif args.dataset == 'cifar10':
        dm = CIFAR10DataModule(
            data_dir=args.cifar10_dir,
            num_workers=args.num_workers,
            normalize=True,
            batch_size=args.batch_size,
            val_split=1000,
        )
        dm.train_transforms = TrainDataTransform(
            image_size=dm.dims[1],
            gaussian_blur=False,
            jitter_strength=0.5,
            final_transforms=dm.default_transforms()
        )
    else:
        raise ValueError(f'--dataset {args.dataset} is not supported')

    model = Image2Vec(
        image_size=dm.dims[-1],
        num_classes=dm.num_classes,
        supervised=args.supervised,
        ae=args.ae,
        ae_latent_dim=args.ae_latent_dim,
        vae=args.vae,
        vae_latent_dim=args.vae_latent_dim,
        simclr=args.simclr,
        lr=args.lr
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.logs_dir, name=''),
        callbacks=[LearningRateMonitor(), LogEmbeddings()],
        accelerator='gpu',
        resume_from_checkpoint=args.checkpoint
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)
