from resnet_pytorch import ResNet, ResidualBlock
from functools import partial

resnet_mini = partial(ResNet, block=ResidualBlock, layers=[2, 2, 2])
resnet18 = partial(ResNet, block=ResidualBlock, layers=[2, 2, 2, 2])
resnet34 = partial(ResNet, block=ResidualBlock, layers=[3, 4, 6, 3])

