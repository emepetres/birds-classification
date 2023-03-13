import os
from pathlib import Path

from fastai.vision.data import (
    IndexSplitter,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    RegexLabeller,
)
from fastai.vision.augment import (
    RandomResizedCrop,
    aug_transforms,
    Normalize,
    imagenet_stats,
)

from fastai.callback import schedule  # noqa: F401
from fastai.vision.learner import vision_learner, accuracy

from birds import config
from birds.utils.kaggle import download_dataset


def get_birds_images(path):
    with open(path / "images.txt", "r") as file:
        lines = [
            path.resolve() / "images" / line.strip().split()[1]
            for line in file.readlines()
        ]
    return lines


def BirdsSplitter(path):
    with open(path / "train_test_split.txt", "r") as file:
        valid_idx = [
            int(line.strip().split()[0]) - 1
            for line in file.readlines()
            if line.strip().split()[1] == "1"
        ]
    return IndexSplitter(valid_idx)


if __name__ == "__main__":
    bs = 64

    if download_dataset(config.OWNER, config.DATASET, config.DATA_PATH):
        import tarfile

        with tarfile.open(Path(config.DATA_PATH) / "CUB_200_2011.tgz", "r:gz") as tar:
            tar.extractall(path=config.DATA_PATH)

        os.remove(Path(config.DATA_PATH) / "CUB_200_2011.tgz")
        os.remove(Path(config.DATA_PATH) / "segmentations.tgz")

    path = Path(config.DATA_PATH) / "CUB_200_2011"

    item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.0, 1.0))
    batch_tfms = [
        *aug_transforms(size=224, max_warp=0),
        Normalize.from_stats(*imagenet_stats),
    ]

    birds = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_birds_images,
        splitter=BirdsSplitter(path),
        get_y=RegexLabeller(pat=r"/([^/]+)_\d+_\d+\.jpg"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    dls = birds.dataloaders(path)

    learner = vision_learner(dls, "vit_tiny_patch16_224", metrics=[accuracy])

    learner.fine_tune(7, base_lr=0.001, freeze_epochs=12)

    learner.export("models/vit_exported")
    learner.save("vit_saved", with_opt=False)
