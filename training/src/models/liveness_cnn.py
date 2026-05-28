"""
LivenessCNN — lightweight liveness detection model.

Classifies faces as:
  0: real (live person)
  1: print attack (photo held up to camera)
  2: screen attack (phone/tablet displaying photo)

Architecture uses depthwise separable convolutions for efficiency.
Input: 112x112x3 RGB face crop
Output: 3-class logits
Size: ~2.8MB (INT8 quantized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LivenessBlock(nn.Module):
    """Depthwise separable conv block for liveness detection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class LivenessCNN(nn.Module):
    """
    Lightweight CNN for face liveness detection.

    Args:
        num_classes: Number of output classes (default: 3)
        dropout_rate: Dropout rate before final layer (default: 0.3)
    """

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # 112x112x3 → 56x56x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 56x56x32 → 56x56x64
            LivenessBlock(32, 64, stride=1),
            # 56x56x64 → 28x28x128
            LivenessBlock(64, 128, stride=2),
            # 28x28x128 → 14x14x256
            LivenessBlock(128, 256, stride=2),
            # 14x14x256 → 7x7x256
            LivenessBlock(256, 256, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes),
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
            Class logits, shape (B, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
