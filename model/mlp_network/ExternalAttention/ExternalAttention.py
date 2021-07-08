import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, drop):
        super(ExternalAttention, self).__init__()
        self.input_size = input_size

        self.Mk = nn.Linear(input_size, hidden_size, bias=False)
        self.Mv = nn.Linear(hidden_size, input_size, bias=False)

        self.softmax = nn.Softmax(dim=-2)
        self.attn_drop = nn.Dropout(drop)

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
        attn = self.Mk(x)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # norm
        x = self.Mv(attn)
        return x
