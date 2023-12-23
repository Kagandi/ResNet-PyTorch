import torch
from PIL import Image, ImageOps, ImageEnhance


class Flip(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        """Flip the image horizontally with a probability p.
        :param p: probability of the image being flipped
        """
        super().__init__()
        self.p = p

    def forward(self, x: Image.Image) -> Image.Image:
        """Flip the image horizontally with a probability p.
        :param x: image to be flipped
        :return: flipped image
        """
        if torch.rand(1) < self.p:
            return ImageOps.flip(x)
        return x


class Rotate(torch.nn.Module):
    def __init__(self, p: float = 0.5, min_degrees: int = 45, max_degrees: int = 90):
        """Rotate the image by a random angle between min_degrees and max_degrees with a probability p.
        :param p: probability of the image being rotated
        :param min_degrees: minimum angle of rotation
        :param max_degrees: maximum angle of rotation
        """
        super().__init__()
        self.p = p
        self.degrees = torch.randint(min_degrees, max_degrees, (1,)).item()

    def forward(self, x: Image.Image) -> Image.Image:
        if torch.rand(1) < self.p:
            return x.rotate(self.degrees)
        return x


class Mirror(torch.nn.Module):
    def __init__(self, p: int = 0.5):
        """Mirror the image horizontally with a probability p.
        :param p: probability of the image being mirrored
        """
        super().__init__()
        self.p = p

    def forward(self, x: Image.Image) -> Image.Image:
        """Mirror the image horizontally with a probability p.
        :param x: image to be mirrored
        :return: mirrored image
        """
        if torch.rand(1) < self.p:
            return ImageOps.mirror(x)
        return x


class Contrast(torch.nn.Module):
    def __init__(
        self, p: float = 0.5, min_factor: float = 0.5, max_factor: float = 1.5
    ):
        """Adjust contrast of the image by a random factor between min_factor and max_factor with a probability p.
        :param p: probability of the image being contrast adjusted
        :param min_factor: minimum contrast adjustment factor
        :param max_factor: maximum contrast adjustment factor
        """
        super().__init__()
        self.p = p
        self.factor = torch.rand(1) * (max_factor - min_factor) + min_factor

    def forward(self, x: Image.Image) -> Image.Image:
        """Adjust contrast of the image by a random factor between min_factor and max_factor with a probability p.
        :param x: image to be contrast adjusted
        :return: contrast adjusted image
        """
        if torch.rand(1) < self.p:
            return ImageEnhance.Contrast(x).enhance(self.factor)
        return x


class Brightness(torch.nn.Module):
    def __init__(
        self, p: float = 0.5, min_factor: float = 0.5, max_factor: float = 1.5
    ):
        """Adjust brightness of the image by a random factor between min_factor and max_factor with a probability p.
        :param p: probability of the image being brightness adjusted
        :param min_factor: minimum brightness adjustment factor
        :param max_factor: maximum brightness adjustment factor
        """
        super().__init__()
        self.p = p
        self.factor = torch.rand(1) * (max_factor - min_factor) + min_factor

    def forward(self, x: Image.Image) -> Image.Image:
        """Adjust brightness of the image by a random factor between min_factor and max_factor with a probability p.
        :param x: image to be brightness adjusted
        :return: brightness adjusted image
        """
        if torch.rand(1) < self.p:
            return ImageEnhance.Brightness(x).enhance(self.factor)
        return x
