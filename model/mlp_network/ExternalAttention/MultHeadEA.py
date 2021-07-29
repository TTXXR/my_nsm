import torch
from torch import nn
from torch.nn import init


class MultHeadEA(nn.Module):
    def __init__(self, input_size, encoder_dim, drop, num_heads, coef):
        super().__init__()
        self.num_heads = num_heads
        self.coef = coef
        self.input_size = input_size
        self.encoder_dim = encoder_dim
        self.k = self.input_size / self.coef  # hidden_size - 64

        self.trans_dim = nn.Linear(self.input_size, self.input_size*self.coef)
        self.num_heads = self.num_heads * self.coef

        self.Mk = nn.Linear(self.input_size * self.coef // self.num_heads, self.k, bias=False)
        self.Mv = nn.Linear(self.k, self.input_size * self.coef // self.num_heads, bias=False)

        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.input_size * self.coef, self.input_size)
        self.proj_drop = nn.Dropout(drop)

        self.softmax = nn.Softmax(dim=-2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.trans_dim(x)
        # x = x.view(x.shape[0], x.shape[1], )

        attn = self.Mk(x)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # norm
        x = self.Mv(attn)
        return x
