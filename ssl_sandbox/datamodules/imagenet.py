from typing import *
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageNet(pl.LightningDataModule):
    def __init__(
            self,
            root: str,
            image_size: Union[int, Tuple[int, int]] = 224,
            scale: Tuple[float, float] = (0.2, 1.0),
            batch_size: int = 32,
            num_workers: int = 0,
    ):
        super().__init__()

        root = self.root = Path(root)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        self.train_dataset = torchvision.datasets.ImageNet(root, transform=transform)
        self.test_dataset = torchvision.datasets.ImageNet(root, split='val', transform=transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """Initializes data loader on each epoch.

        Returns:
            DataLoader: loads batch of images and labels.
                Labels can be then discarded if not needed (e.g. at pre-training phase).
        """
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataloader, self.batch_size, num_workers=self.num_workers)
