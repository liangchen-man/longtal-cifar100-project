from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ResNetCifar(nn.Module):
    """ResNet for 32x32 images following the CIFAR depth rule 6n+2."""

    def __init__(self, blocks_per_stage: int = 5, num_classes: int = 100):
        super().__init__()
        self.in_channels = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for block_stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x):
        return self.fc(self.forward_features(x))


def _blocks_from_name(name: str) -> int:
    name = name.lower()
    if name in {"resnet20", "resnet-20"}:
        return 3
    if name in {"resnet32", "resnet-32"}:
        return 5
    if name in {"resnet44", "resnet-44"}:
        return 7
    if name in {"resnet56", "resnet-56"}:
        return 9
    raise ValueError(f"Unsupported CIFAR ResNet variant: {name}")


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "resnet32")
    num_classes = int(model_cfg.get("num_classes", 100))
    return ResNetCifar(blocks_per_stage=_blocks_from_name(name), num_classes=num_classes)
