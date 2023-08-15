from torch import nn

import pytorch_lightning as pl


class Sensemble(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            num_prototypes: int,
    ):
        super().__init__()