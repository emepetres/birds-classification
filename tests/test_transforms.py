import pytest

from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from fastai.vision.data import PILImage
import fastai.vision.augment as fastai_aug

from deployment.transforms import CenterCropPad

DATA_PATH = "data/kaggle/200-bird-species-with-11788-images"


def get_birds_images(path: Path) -> List[str]:
    with open(path / "images.txt", "r") as file:
        lines = [
            path.resolve() / "images" / line.strip().split()[1]
            for line in file.readlines()
        ]
    return lines


class TestTransforms:
    im_idx = 0

    @pytest.fixture
    def img_paths(self) -> List[str]:
        path = Path(DATA_PATH) / "CUB_200_2011"
        return get_birds_images(path.resolve())

    @pytest.fixture
    def im_fastai(self, img_paths) -> PILImage:
        fname = img_paths[self.im_idx]
        return PILImage.create(fname)

    @pytest.fixture
    def im_pil(self, img_paths) -> Image:
        fname = img_paths[self.im_idx]
        return Image.open(fname)

    def testImageFastaiEqualsPillow(self, im_fastai, im_pil):
        assert (np.array(im_fastai) == np.array(im_pil)).all()

    def testCropPadFastaiEqualsTorch(self, im_fastai, im_pil):
        crop_fastai = fastai_aug.CropPad((460, 460))
        crop_torch = CenterCropPad((460, 460))

        assert (np.array(crop_fastai(im_fastai)) == np.array(crop_torch(im_pil))).all()
