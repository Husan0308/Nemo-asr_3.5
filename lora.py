import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        linear,
        r=16,
        alpha=32,
        dropout=0.05,
    ):
        super().__init__()

        self.linear = linear

        in_features = linear.in_features
        out_features = linear.out_features

        self.r = r
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout)

        linear.weight.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(r, in_features)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r)
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5)
        )

    def forward(self, x):

        result = self.linear(x)

        lora = (
            self.dropout(x)
            @ self.lora_A.t()
        ) @ self.lora_B.t()

        return result + (
            lora * self.scaling
        )