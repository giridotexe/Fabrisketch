"""
Fabrisketch Multi-Head U-Net Model (`ml/unet_model.py`)
======================================================
Takes a noisy user PNG image (3 channels: R, G, B) and predicts three outputs:
  1. `stitch_logits`: (B, 4, H, W) -> [Background, Tatami Fill, Satin Column, Running Stitch]
  2. `angle_vec`:     (B, 2, H, W) -> Unit vector [cos(2*theta), sin(2*theta)] for thread direction
  3. `corner_prob`:   (B, 1, H, W) -> Probability map of sharp corners & region boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EmbroideryDigitizerUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_filters: int = 32, num_classes: int = 4):
        super().__init__()
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.bottleneck = DoubleConv(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool2d(2)

        # Decoder (Upsampling with skip connections)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)

        # 3 Task-Specific Prediction Heads
        self.class_head = nn.Conv2d(base_filters, num_classes, kernel_size=1)
        self.angle_head = nn.Conv2d(base_filters, 2, kernel_size=1)
        self.corner_head = nn.Conv2d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        stitch_logits = self.class_head(d1)
        angle_vec = F.normalize(self.angle_head(d1), p=2, dim=1, eps=1e-6)
        corner_prob = torch.sigmoid(self.corner_head(d1))

        return stitch_logits, angle_vec, corner_prob