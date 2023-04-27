# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import torch
from torch import nn


class Classify2D(nn.Module):
    def __init__(
        self,
        input_size=(1, 28, 28),
        lin_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.lin_size = lin_size

        from src.models.components.resnet import VGG_FeatureExtractor

        # self.backbone = backbone
        self.backbone = VGG_FeatureExtractor(1, 128)
        ch = self.backbone(torch.randn(input_size)).view(-1).size()
        self.dense = nn.Linear(ch[0], self.lin_size)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.backbone(x)
        x = x.view(batch_size, -1)

        x = self.dense(x)

        return x
