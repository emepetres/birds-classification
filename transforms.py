import math
from typing import Union, Tuple

import torch

import torch.nn.functional as F
import torchvision.transforms.functional as tvf

# import torchvision.transforms as tvtfms
# # import operator as op
from PIL import Image

# # from torch import nn
# # from timm import create_model


def crop(image: Union[Image.Image, torch.tensor], size: Tuple[int, int]) -> Image:
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
    top = (image.size[-2] - size[0]) // 2
    left = (image.size[-1] - size[1]) // 2

    top = max(top, 0)
    left = max(left, 0)

    height = min(top + size[0], image.size[-2])
    width = min(left + size[1], image.size[-1])
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
    top = (image.size[-2] - size[0]) // 2
    left = (image.size[-1] - size[1]) // 2

    pad_top = max(-top, 0)
    pad_left = max(-left, 0)

    height, width = (
        max(size[1] - image.size[-2] + top, 0),
        max(size[0] - image.size[-1] + left, 0),
    )
    return tvf.pad(image, [pad_top, pad_left, height, width], padding_mode="constant")


def resized_crop_pad(
    image: Union[Image.Image, torch.tensor],
    size: Tuple[int, int],
    extra_crop_ratio: float = 0.14,
) -> Image:
    """
    Takes a `PIL.Image`, resize it according to the
    `extra_crop_ratio`, and then crops and pads
    it to `size`.

    Args:
        image (`PIL.Image`):
            An image to perform padding on
        size (`tuple` of integers):
            A size to crop and pad to, should be in the form
            of (width, height)
        extra_crop_ratio (float):
            The ratio of size at the edge cropped out. Default 0.14
    """

    maximum_space = max(size[0], size[1])
    extra_space = maximum_space * extra_crop_ratio
    extra_space = math.ceil(extra_space / 8) * 8
    extended_size = (size[0] + extra_space, size[1] + extra_space)
    resized_image = image.resize(extended_size, resample=Image.Resampling.BILINEAR)

    if extended_size != size:
        resized_image = pad(crop(resized_image, size), size)

    return resized_image


def gpu_crop(batch: torch.tensor, size: Tuple[int, int]):
    """
    Crops each image in `batch` to a particular `size`.

    Args:
        batch (array of `torch.Tensor`):
            A batch of images, should be of shape `NxCxWxH`
        size (`tuple` of integers):
            A size to pad to, should be in the form
            of (width, height)

    Returns:
        A batch of cropped images
    """
    # Split into multiple lines for clarity
    affine_matrix = torch.eye(3, device=batch.device).float()
    affine_matrix = affine_matrix.unsqueeze(0)
    affine_matrix = affine_matrix.expand(batch.size(0), 3, 3)
    affine_matrix = affine_matrix.contiguous()[:, :2]

    coords = F.affine_grid(affine_matrix, batch.shape[:2] + size, align_corners=True)

    top_range, bottom_range = coords.min(), coords.max()
    zoom = 1 / (bottom_range - top_range).item() * 2

    resizing_limit = (
        min(batch.shape[-2] / coords.shape[-2], batch.shape[-1] / coords.shape[-1]) / 2
    )

    if resizing_limit > 1 and resizing_limit > zoom:
        batch = F.interpolate(
            batch,
            scale_factor=1 / resizing_limit,
            mode="area",
            recompute_scale_factor=True,
        )
    return F.grid_sample(
        batch, coords, mode="bilinear", padding_mode="reflection", align_corners=True
    )
