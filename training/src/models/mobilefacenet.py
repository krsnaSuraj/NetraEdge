"""
MobileFaceNet — lightweight face recognition model.

Architecture based on:
- "MobileFaceNets: Efficient CNNs for Real-Time Face Verification" (2018)
- Modified with depthwise separable convolutions and squeeze-excitation blocks
- Knowledge distillation from ArcFace-R100 teacher

Input: 112x112x3 RGB face crop
Output: 128-d L2-normalized embedding
Size: ~4.8MB (INT8 quantized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution — 8-9x fewer FLOPs than standard conv."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class SqueezeExcitation(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class MobileBlock(nn.Module):
    """Mobile network block: expansion → depthwise-sep → SE → projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        mid_channels = in_channels * 2
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = DepthwiseSeparableConv(mid_channels, out_channels, stride)
        self.se = SqueezeExcitation(out_channels)
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        if self.use_residual:
            out = out + x
        return out


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet for face recognition.

    Args:
        embedding_dim: Output embedding dimension (default: 128)
        use_attention: Enable squeeze-excitation blocks (default: True)
    """

    def __init__(self, embedding_dim: int = 128, use_attention: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            MobileBlock(64, 64, stride=1),
            MobileBlock(64, 128, stride=2),
            MobileBlock(128, 128, stride=1),
            MobileBlock(128, 128, stride=1),
            MobileBlock(128, 256, stride=2),
            MobileBlock(256, 256, stride=1),
            MobileBlock(256, 256, stride=1),
            MobileBlock(256, 256, stride=1),
            MobileBlock(256, 256, stride=1),
            MobileBlock(256, 512, stride=2),
            MobileBlock(512, 512, stride=1),
            MobileBlock(512, 512, stride=1),
        ) if use_attention else nn.Sequential(
            DepthwiseSeparableConv(64, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512, stride=1),
        )

        self.final = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Face crops, shape (B, 3, 112, 112), values in [0, 1]

        Returns:
            L2-normalized embeddings, shape (B, embedding_dim)
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_size_mb(self) -> float:
        param_bytes = sum(p.numel() * 4 for p in self.parameters())
        return param_bytes / (1024 * 1024)
