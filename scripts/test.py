from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm

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
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)

    return parser.parse_args()


def collect_data(model, dataloader, desc):
    data = []
    for batch in tqdm(dataloader, desc=desc):
        images, labels = batch
        vecs = model(images.to(model.device))
        qq_logits = model.qq_head(vecs)
        data.extend([
            {'vec': v.tolist(), 'embedding': q.tolist(), 'label': str(l.item())}
            for v, q, l in zip(vecs, qq_logits, labels)
        ])
    return data


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.cifar10_dir,
        num_workers=args.num_workers,
        normalize=True,
        batch_size=args.batch_size,
        val_split=1000,
    )
    dm.prepare_data()
    dm.setup()

    logs_dir = Path(args.logs_dir)

    ckpt,  = (logs_dir / 'checkpoints').iterdir()
    model = Image2Vec.load_from_checkpoint(ckpt)
    model.to(args.device)
    model.eval()

    with open(logs_dir / 'train_data.json', 'w') as f:
        json.dump(collect_data(model, dm.train_dataloader(), desc='train'), f)
    with open(logs_dir / 'val_data.json', 'w') as f:
        json.dump(collect_data(model, dm.val_dataloader(), desc='val'), f)
    with open(logs_dir / 'test_data.json', 'w') as f:
        json.dump(collect_data(model, dm.test_dataloader(), desc='test'), f)


if __name__ == '__main__':
    main(parse_args())
