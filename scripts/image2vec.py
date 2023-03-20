from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from pl_bolts.datamodules import CIFAR10DataModule

from ssl_sandbox.models.image2vec import Image2Vec, LogEmbeddings
from ssl_sandbox.transforms import SimCLRViews, BYOLViews


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--cifar10_dir')
    parser.add_argument('--logs_dir')

    parser.add_argument('--supervised', default=False, action='store_true')
    parser.add_argument('--ae', default=False, action='store_true')
    parser.add_argument('--ae_latent_dim', type=int, default=128)
    parser.add_argument('--vae', default=False, action='store_true')
    parser.add_argument('--vae_latent_dim', type=int, default=128)
    parser.add_argument('--simclr', default=False, action='store_true')
    parser.add_argument('--vicreg', default=False, action='store_true')
    parser.add_argument('--qq', default=False, action='store_true')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoint')

    return parser.parse_args()


def main(args):
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule(
            data_dir=args.cifar10_dir,
            num_workers=args.num_workers,
            normalize=True,
            batch_size=args.batch_size,
            val_split=1000,
        )
        image_size = 32
        params = dict(first_conv=False, maxpool1=False)
    else:
        raise ValueError(f'--dataset {args.dataset} is not supported')
    
    if args.simclr:
        simclr_views = SimCLRViews(size=image_size, blur=False, jitter_strength=0.5, final_transforms=dm.default_transforms())
        dm.train_transforms = simclr_views.train_transforms
    elif args.vicreg:
        byol_views = BYOLViews(size=32, final_transforms=dm.default_transforms())
        dm.train_transforms = byol_views.train_transforms
    
    model = Image2Vec(
        image_size=image_size,
        num_classes=dm.num_classes,
        supervised=args.supervised,
        ae=args.ae,
        ae_dim=args.ae_latent_dim,
        vae=args.vae,
        vae_dim=args.vae_latent_dim,
        simclr=args.simclr,
        vicreg=args.vicreg,
        qq=args.qq,
        **params,
        lr=args.lr
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.logs_dir, name=''),
        callbacks=[LearningRateMonitor(), LogEmbeddings()],
        accelerator='gpu',
        max_epochs=args.num_epochs,
        resume_from_checkpoint=args.checkpoint
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    args = parse_args()
    main(args)
