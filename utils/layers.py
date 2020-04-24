from torch import nn


class NoOpt(nn.Module):
    def forward(self, x):
        return x
