import torch.nn as nn
from .attention.gelu import GELU
from .ExternalAttention import ExternalAttention


class Encoder(nn.Module):
    def __init__(self, dim, mlp_ratio, layer_num, dropout, act=GELU, norm=nn.LayerNorm):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.norm = norm(dim)

        # pure MLP
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act, drop=dropout)

        # external attention
        self.attn_num = layer_num
        self.attn_blocks = nn.ModuleList(
            [ExternalAttention(input_size=dim, hidden_size=int(dim * mlp_ratio), drop=dropout)
             for _ in range(self.attn_num)])

    def forward(self, x):
        # pure MLP
        # x = x + self.drop(self.mlp1(self.norm(x)))

        # external attention
        for attn in self.attn_blocks:
            x = x + self.drop(attn(self.norm(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=GELU, drop=0.3):
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
