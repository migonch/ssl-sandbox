from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from pl_bolts.datamodules import CIFAR10DataModule

from ssl_sandbox.models.image2vec import Image2Vec, LogEmbeddings, QQTeacherUpdate
from ssl_sandbox.transforms import SimCLRViews, BYOLViews


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cifar10_dir')
    parser.add_argument('--logs_dir')

    parser.add_argument('--supervised', default=False, action='store_true')
    parser.add_argument('--ssl_method', default='qq')
    parser.add_argument('--qq_reg_weight', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=300)

    parser.add_argument('--checkpoint')

    return parser.parse_args()


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.cifar10_dir,
        num_workers=args.num_workers,
        normalize=True,
        batch_size=args.batch_size,
        val_split=1000,
    )
    image_size = 32
    blur = False
    jitter_strength = 0.5
    architecture_params = dict(first_conv=False, maxpool1=False)

    assert args.ssl_method in ['ae', 'vae', 'simclr', 'vicreg', 'qq', 'none']
    if args.ssl_method in ['simclr', 'vicreg']:
        dm.train_transforms = SimCLRViews(image_size, jitter_strength, blur, final_transforms=dm.default_transforms())
    elif args.ssl_method == 'qq':
        dm.train_transforms = BYOLViews(image_size, final_transforms=dm.default_transforms())

    model = Image2Vec(
        image_size=image_size,
        num_classes=dm.num_classes,
        supervised=args.supervised,
        ssl_method=args.ssl_method,
        qq_reg_weight=args.qq_reg_weight,
        **architecture_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs
    )
    callbacks = [LearningRateMonitor(), LogEmbeddings()]
    if args.ssl_method == 'qq':
        callbacks.append(QQTeacherUpdate())
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.logs_dir, name=''),
        callbacks=callbacks,
        accelerator='gpu',
        max_epochs=args.num_epochs,
        resume_from_checkpoint=args.checkpoint
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main(parse_args())
