import pytest

from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from fastai.vision.data import PILImage
import fastai.vision.augment as fastai_aug

from deployment.transforms import resized_crop_pad

DATA_PATH = "data/kaggle/200-bird-species-with-11788-images"


def get_birds_images(path: Path) -> List[str]:
    with open(path / "images.txt", "r") as file:
        lines = [
            path.resolve() / "images" / line.strip().split()[1]
            for line in file.readlines()
        ]
    return lines


class TestTransforms:
    im_idx = 510

    @pytest.fixture
    def img_paths(self) -> List[str]:
        path = Path(DATA_PATH) / "CUB_200_2011"
        return get_birds_images(path.resolve())

    @pytest.fixture
    def im_fastai(self, img_paths: List[str]) -> PILImage:
        fname = img_paths[self.im_idx]
        return PILImage.create(fname)

    @pytest.fixture
    def im_pil(self, img_paths: List[str]) -> Image:
        fname = img_paths[self.im_idx]
        return Image.open(fname)

    def testImageFastaiEqualsPillow(self, im_fastai: PILImage, im_pil: Image):
        assert (np.array(im_fastai) == np.array(im_pil)).all()

    # RandomResizedCrop is not exactly equal to CropPad in validation
    # # def testRandomResizedCropEqualsCropPadInValidation(self, im_fastai: PILImage):
    # #     crop_fastai = fastai_aug.CropPad((460, 460))
    # #     crop_rrc = fastai_aug.RandomResizedCrop((460, 460))

    # #     cropped_rrc = crop_rrc(im_fastai, split_idx=1)
    # #     cropped_fastai = crop_fastai(im_fastai, split_idx=1)

    # #     assert (np.array(cropped_rrc) == np.array(cropped_fastai)).all()

    def testRandomResizedCropInValidationEqualsCustomResizedCropPad(
        self, im_fastai: PILImage, im_pil: Image
    ):
        crop_rrc = fastai_aug.RandomResizedCrop((460, 460))

        assert (
            np.array(crop_rrc(im_fastai, split_idx=1))
            == np.array(resized_crop_pad(im_pil, (460, 460)))
        ).all()
