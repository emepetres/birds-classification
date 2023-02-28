from typing import Literal
import torchvision.transforms as tvtfms


def CenterCropPad(size: tuple[Literal[460], Literal[460]]):
    return tvtfms.CenterCrop(size)
