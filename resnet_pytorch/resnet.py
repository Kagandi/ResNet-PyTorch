import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, downsample=None
    ):
        """Residual block of ResNet.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: stride of the convolutional layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.
        :param x: input to the residual block
        :return: output of the residual block
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: list,
        in_channels: int = 64,
        num_classes: int = 10,
        stride: int = 2,
    ):
        """ResNet model.
        :param block: residual block to be used
        :param in_channels: number of input channels
        :param layers: number of residual blocks in each layer
        :param num_classes: number of output classes
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(3, in_channels, 7, padding=1, stride=2, bias=False)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layers = [self.make_layer(block, in_channels, layers[0])] + [
            self.make_layer(block, in_channels * 2**i, layers[i], stride)
            for i in range(1, len(layers))
        ]

        self.fwd = nn.Sequential(
            *self.layers,
        )

        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Linear(in_channels * (2 ** (len(layers) - 1)), num_classes)

    def make_layer(
        self, block: nn.Module, out_channels: int, blocks: list, stride: int = 1
    ):
        """Creates a layer of residual blocks.
        :param block: residual block to be used
        :param out_channels: number of output channels
        :param blocks: number of residual blocks in the layer
        :param stride: stride of the convolutional layers
        :return: a layer of residual blocks
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet model.
        :param x: input to the ResNet model
        :return: output of the ResNet model
        """
        x = self.max_pool(self.relu(self.bn(self.conv(x))))
        x = self.fwd(x)
        x = self.avg_pool(x)

        x = self.fc(x)
        return x


def conv_block(in_channels: int, out_channels: int, pool=False) -> nn.Sequential:
    """Convolutional block of two convolutional layers followed by a max pooling layer.
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param pool: whether to apply max pooling or not
    :return: convolutional block
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels: int = 64, num_classes: int = 10):
        """ResNet9 model.
        :param in_channels: number of input channels
        :param num_classes: number of output classes
        """
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        """Forward pass of the ResNet9 model.
        :param xb: input to the ResNet9 model
        :return: output of the ResNet9 model
        """
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
