from torch import nn
from torch.backends import cudnn


class CTCLoss(nn.CTCLoss):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super().__init__(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        with cudnn.flags(enabled=False):
            return super().forward(log_probs=log_probs, targets=targets,
                                   input_lengths=input_lengths, target_lengths=target_lengths)
