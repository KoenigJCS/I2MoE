import torch
import torch.nn as nn


class SimpleSleepCNN(nn.Module):
    """A small, robust 2D-CNN for spectrogram classification.

    Input: x shaped (B, M, F, T) where M=modalities, F=freq bins, T=time frames.
    This model treats the modality axis as channels for Conv2d and applies a
    few Conv-BN-ReLU-Pool blocks followed by adaptive pooling and an MLP head.
    """

    def __init__(self, num_modalities: int, freq_bins: int, num_classes: int, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        in_ch = num_modalities
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # collapse spatial dims to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # small init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias) # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, M, F, T)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Expect input (B, M, F, T). Conv2d expects (B, C, H, W) where H=F, W=T
        out = self.encoder(x)
        out = self.dropout(out)
        logits = self.head(out)
        return logits


if __name__ == '__main__':
    # quick smoke test
    model = SimpleSleepCNN(num_modalities=1, freq_bins=129, num_classes=5)
    dummy = torch.randn(2, 1, 129, 10)
    out = model(dummy)
    print('SimpleSleepCNN smoke OK, output shape:', out.shape)
