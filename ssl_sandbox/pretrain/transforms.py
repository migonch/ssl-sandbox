from typing import *
from PIL import Image

import torch
import torchvision.transforms as T


SIMCLR_COLOR_JITTER_PARAMS = dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
BYOL_COLOR_JITTER_PARAMS = dict(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)


class RandomView:
    """Standard augmentations reused in the most of SSL methods.
    The default parameters are the same as in SimCLR (https://arxiv.org/abs/2002.05709).
    """
    def __init__(
            self,
            size: int,
            scale: Tuple[float, float] = (0.08, 1.0),
            flip_p: float = 0.5,
            color_jitter_p: float = 0.8,
            brightness: float = 0.8,
            contrast: float = 0.8,
            saturation: float = 0.8,
            hue: float = 0.2,
            grayscale_p: float = 0.2,
            blur_p: float = 0.5,
            solarization_p: float = 0.0,
            final_transforms: Optional[Callable] = None
    ) -> None:
        color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        augmentations = [
            T.RandomResizedCrop(size, scale, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(flip_p),
            T.RandomApply([color_jitter], color_jitter_p),
            T.RandomGrayscale(grayscale_p),
        ]
        if blur_p > 0:
            kernel_size = int(0.1 * size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            augmentations.append(T.RandomApply([T.GaussianBlur(kernel_size)], blur_p))
        if solarization_p > 0:
            augmentations.append(T.RandomSolarize(0.5, solarization_p))

        augmentations = T.Compose(augmentations)
        if final_transforms is None:
            final_transforms = T.ToTensor()

        self.augmentations = augmentations
        self.final_transforms = final_transforms

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Any:
        return self.final_transforms(self.augmentations(image))


class SimCLRViews:
    """Set ``blur=False`` for CIFAR10 dataset (according to https://arxiv.org/abs/2002.05709).
    """
    def __init__(
            self,
            size: int,
            scale: Tuple[float, float] = (0.08, 1.0),
            jitter_strength: float = 1.0,
            blur: bool = True,
            final_transforms: Optional[Callable] = None,
            views_number: int = 2
    ) -> None:
        jitter_params = {k: v * jitter_strength for k, v in SIMCLR_COLOR_JITTER_PARAMS.items()}
        blur_p = 0.5 if blur else 0.0
        if final_transforms is None:
            final_transforms = T.ToTensor()

        self.random_view = RandomView(size, scale, **jitter_params, blur_p=blur_p,
                                      final_transforms=final_transforms)
        self.final_transforms = final_transforms
        self.views_number = views_number

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Any:
        return (
            self.final_transforms(image),
            *[self.random_view(image) for _ in range(self.views_number)]
        )


class BYOLViews:
    def __init__(
            self,
            size: int,
            final_transforms: Optional[Callable] = None
    ) -> None:
        if final_transforms is None:
            final_transforms = T.ToTensor()

        self.online_view = RandomView(
            size, **BYOL_COLOR_JITTER_PARAMS, blur_p=1.0,
            final_transforms=final_transforms
        )
        self.target_view = RandomView(
            size, **BYOL_COLOR_JITTER_PARAMS, blur_p=0.1, solarization_p=0.2,
            final_transforms=final_transforms
        )
        self.final_transforms = final_transforms

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Any:
        return (
            self.final_transforms(image),
            self.online_view(image),
            self.target_view(image)
        )


class MultiCrop:
    """Originally proposed in SwAV (https://arxiv.org/abs/2006.09882).
    Also used in DINO (https://arxiv.org/abs/2104.14294).
    The hyperparameters are taken from DINO paper and from
    https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/main_dino.py#L419.
    """
    def __init__(
            self,
            global_views_size: int = 224,
            local_views_size: int = 96,
            global_views_scale: Tuple[float, float] = (0.3, 1.0),
            local_views_scale: Tuple[float, float] = (0.05, 0.3),
            local_views_number: int = 6,
            final_transforms: Optional[Callable] = None
    ) -> None:
        if final_transforms is None:
            final_transforms = T.ToTensor()

        self.first_global_view = RandomView(
            global_views_size, global_views_scale,
            **BYOL_COLOR_JITTER_PARAMS, blur_p=1.0,
            final_transforms=final_transforms
        )
        self.second_global_view = RandomView(
            global_views_size, global_views_scale,
            **BYOL_COLOR_JITTER_PARAMS, blur_p=0.1, solarization_p=0.2,
            final_transforms=final_transforms
        )
        self.local_view = RandomView(
            local_views_size, local_views_scale,
            **BYOL_COLOR_JITTER_PARAMS, blur_p=0.5,
            final_transforms=final_transforms
        )
        self.final_transforms = final_transforms
        self.local_views_number = local_views_number

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Any:
        return (
            self.final_transforms(image),
            self.first_global_view(image),
            self.second_global_view(image),
            *[self.random_local_view(image) for _ in range(self.local_views_number)]
        )
