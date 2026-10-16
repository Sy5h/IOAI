"""Torch-only submission module for IOAI Field.

Copy this file to custom_model.py beside the official notebook before training.
Every persistent tensor is a Parameter; there are no registered buffers.
"""

import torch
import torch.nn as nn


class DensePart(nn.Module):
    def __init__(self, output_dim, width=64, depth=3):
        super().__init__()
        layers = []
        input_dim = 2
        for _ in range(depth):
            layers.extend([nn.Linear(input_dim, width), nn.GELU()])
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class DropoutField(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(4, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[128., 256., 512., 1024.]]))
            self.linear.bias.fill_(-1920.)
        for parameter in self.linear.parameters():
            parameter.requires_grad_(False)

    def forward(self, xy):
        ones = torch.ones_like(xy[:, :1]).expand(-1, 4)
        return self.linear(self.dropout(ones))


class CustomModel(nn.Module):
    I, O, A, I_ENTROPY, BG = range(5)

    def __init__(self, i_min=1e10, i_max=2e10, width=64, depth=3):
        super().__init__()
        self.router = DensePart(5, width, depth)
        self.i_part = nn.Linear(2, 1)
        self.o_part = DensePart(1, width, depth)
        self.last_i_part = DropoutField()
        self.i_scale = nn.Parameter(torch.tensor([float(i_min), float(i_max - i_min)]),
                                    requires_grad=False)
        # Scalar metadata is permitted by the official validator.
        self.i_threshold = 0.0

    def regions(self, xy):
        logits = self.router(xy)
        region = logits.argmax(dim=1)
        other = logits.clone()
        other[:, self.I] = -torch.inf
        competitor_value, competitor_class = other.max(dim=1)
        trusted_i = logits[:, self.I] - competitor_value >= self.i_threshold
        return torch.where((region == self.I) & ~trusted_i, competitor_class, region)

    def forward(self, xy):
        region = self.regions(xy)
        output = torch.zeros_like(xy[:, :1])
        first_i = region == self.I
        if first_i.any():
            normalized = self.i_part(xy[first_i]).clamp(0.0, 1.0)
            output[first_i] = self.i_scale[0] + self.i_scale[1] * normalized
        letter_o = region == self.O
        if letter_o.any():
            output[letter_o] = torch.tanh(self.o_part(xy[letter_o]))
        output[region == self.A] = -1.0
        entropy_i = region == self.I_ENTROPY
        if entropy_i.any():
            output[entropy_i] = self.last_i_part(xy[entropy_i])
        return output
