import torch
from torch import nn


class ResidualBlock(nn.Module):
    expansion: int = 1

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
        self.layer1 = self.make_layer(block, in_channels, layers[0])
        self.layer2 = self.make_layer(block, in_channels * 2, layers[1], 2)
        self.layer3 = self.make_layer(block, in_channels * 4, layers[2], 2)
        self.layer4 = self.make_layer(block, in_channels * 8, layers[3], 2)
        self.avg_pool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels * 8 * block.expansion, num_classes)

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x
