import torch.nn as nn

from timm.models.resnet import (
    BasicBlock as _BasicBlock,
    Bottleneck as _Bottleneck,
    ResNet,
    _create_resnet
)


class BasicBlock(_BasicBlock):
    def __init__(
            self,
            *args,
            dropout_rate: float = 0.0,
            drop_channel_rate: float = 0.0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(dropout_rate)
        self.drop_channel = nn.Dropout2d(drop_channel_rate)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dropout(x)
        x = self.drop_channel(x)
        x = self.drop_block(x)

        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(_Bottleneck):
    def __init__(
            self,
            *args,
            dropout_rate: float = 0.0,
            drop_channel_rate: float = 0.0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(dropout_rate)
        self.drop_channel = nn.Dropout2d(drop_channel_rate)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.dropout(x)
        x = self.drop_channel(x)
        x = self.drop_block(x)

        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def resnet50(dropout_rate=0.0, drop_channel_rate=0.0, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    if 'block_args' in kwargs:
        kwargs['block_args'].update(dropout_rate=dropout_rate, drop_channel_rate=drop_channel_rate)
    else:
        kwargs['block_args'] = dict(dropout_rate=dropout_rate, drop_channel_rate=drop_channel_rate)

    return _create_resnet('resnet50', block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def adapt_to_cifar10(resnet: ResNet):
    """See https://arxiv.org/pdf/2002.05709.pdf, Appendix B.9.
    """
    resnet.conv1 = nn.Conv2d(resnet.conv1.in_channels, resnet.conv1.out_channels,
                             kernel_size=3, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    return resnet
