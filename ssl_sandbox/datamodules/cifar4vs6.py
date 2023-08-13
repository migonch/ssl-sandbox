from typing import *

from torch.utils.data import Subset

from pl_bolts.datamodules import CIFAR10DataModule


class CIFAR4vs6DataModule(CIFAR10DataModule):
    ood_classes = [0, 1, 3, 5]  # out-of-distribution classes: airoplane, automobile, cat, dog
    id_classes = sorted(set(range(10)).difference(ood_classes))  # in-distribution classes
    EXTRA_ARGS = {
        'target_transform': lambda label: (
            CIFAR4vs6DataModule.id_classes.index(label)
            if label in CIFAR4vs6DataModule.id_classes
            else -1
        )
    }  # these kwargs are given to torchvision.datasets.CIFAR10 class

    @property
    def num_classes(self) -> int:
        return len(self.id_classes)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)

        if stage == "fit" or stage is None:
            self.dataset_train = Subset(
                dataset=self.dataset_train.dataset,
                indices=[i for i in self.dataset_train.indices
                         if self.dataset_train.dataset.targets[i] in CIFAR4vs6DataModule.id_classes]
            )
