from typing import Literal, Union, Tuple

import torch

# # import torch.nn.functional as F
import torchvision.transforms.functional as tvf

# import torchvision.transforms as tvtfms
# # import operator as op
from PIL import Image

# # from torch import nn
# # from timm import create_model


def center_crop(
    image: Union[Image.Image, torch.tensor], size: Tuple[int, int]
) -> Image:
    """
    Takes a `PIL.Image` and crops it `size` unless one
    dimension is larger than the actual image. Padding
    must be performed afterwards if so.

    Args:
        image (`PIL.Image`):
            An image to perform cropping on
        size (`tuple` of integers):
            A size to crop to, should be in the form
            of (width, height)

    Returns:
        An augmented `PIL.Image`
    """
    top = (image.shape[-1] - size[0]) // 2
    left = (image.shape[-2] - size[1]) // 2

    top = max(top, 0)
    left = max(left, 0)

    height = min(top + size[0], image.shape[-1])
    width = min(left + size[1], image.shape[-2])
    return image.crop((top, left, height, width))


def pad(image, size: Tuple[int, int]) -> Image:
    """
    Takes a `PIL.Image` and pads it to `size` with
    zeros.

    Args:
        image (`PIL.Image`):
            An image to perform padding on
        size (`tuple` of integers):
            A size to pad to, should be in the form
            of (width, height)

    Returns:
        An augmented `PIL.Image`
    """
    top = (image.shape[-1] - size[0]) // 2
    left = (image.shape[-2] - size[1]) // 2

    pad_top = max(-top, 0)
    pad_left = max(-left, 0)

    height, width = (
        max(size[1] - image.shape[-1] + top, 0),
        max(size[0] - image.shape[-2] + left, 0),
    )
    return tvf.pad(image, [pad_top, pad_left, height, width], padding_mode="constant")


def CenterCropPad(size: tuple[Literal[460], Literal[460]]):
    # return tvtfms.CenterCrop(size)
    def _crop_pad(img):
        return pad(center_crop(img, size), size)

    return _crop_pad
