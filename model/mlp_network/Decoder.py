import torch.nn as nn
from .attention.gelu import GELU


class Decoder(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(dims[0], dims[1]),
                                   GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(dims[1], dims[2]),
                                   GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(dims[2], dims[3]),
                                   GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(dims[3], dims[4]),
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x
