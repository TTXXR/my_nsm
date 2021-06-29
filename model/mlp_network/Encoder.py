import torch.nn as nn
from .attention.gelu import GELU


class Encoder(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout, act=GELU,
                 norm=nn.LayerNorm):  # 197 = 16**2 + 1
        super().__init__()

        self.drop = nn.Dropout(dropout)
        # FF over features
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act, drop=dropout)
        self.norm1 = norm(dim)
        # FF over patches

    def forward(self, x):
        x = x + self.drop(self.mlp1(self.norm1(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
