import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from inpainting.masked_img_dataset import MaskedImageDataset
from inpainting.params import Params


def get_bigtex_dataloader(split_name: str, params: Params) -> DataLoader:
    """
    Builds a Big Texture dataloader, which samples one or more large images to create training images.

    :param split_name: must be one of ['train', 'val', 'test']
    :param params: params obj
    :return: dataloader
    """
    assert split_name in ['train', 'val', 'test'], f'Unknown split name: {split_name}'
    bigtex_data = BigTextureDataset(params.data_dir_path, params.initial_sample_size, params.img_size, params.samples_per_epoch)
    dataset = MaskedImageDataset(bigtex_data, params.img_size, params.mask_size, transform=params.get_transform(), augs_per_example=params.augs_per_example, use_wandering_mask=params.wandering_mask)

    dl = DataLoader(dataset,
                    batch_size=params.batch_size,
                    num_workers=params.train_worker_count,
                    persistent_workers=True
                    )
    return dl


class BigTextureDataset(Dataset):
    """
    Dataset class that takes random samples from one (or many) large source images.
    """

    def __init__(self, img_dir_path: Path, initial_sample_size: int, final_sample_size: int, total_sample_count: int):
        """
        Creates a new sampling dataset.

        :param img_dir_path: Path directory containing the source image(s)
        :param initial_sample_size: size of image sampled from source before transforms are applied
        :param final_sample_size: final sampled image size
        :param total_sample_count: dataset length; how many samples to randomly draw from the source images
        """
        self.data_path = img_dir_path
        self.imgs = []  # list of tuples in format (img, (max_row, max_col))
        self.initial_sample_size = initial_sample_size
        self.final_sample_size = final_sample_size
        self.sample_count = total_sample_count
        self.file_count = 0
        self.filenames = []
        self._load_files()

    def _load_files(self):
        """
        Preloads all images into memory (because there should be only a small number of them).

        :return: None
        """
        filenames = [name for name in os.listdir(self.data_path) if os.path.isfile(str(self.data_path / name))]
        self.imgs = []

        for filename in filenames:
            img = Image.open(self.data_path / filename).copy()
            img = self._source_transform(img)
            if img.shape[0] == 4:  # remove alpha channel if it exists
                img = img[:-1]

            # note: img.size returns (width, height)
            max_sample_row = img.shape[1] - self.initial_sample_size
            max_sample_col = img.shape[2] - self.initial_sample_size
            self.imgs.append((img, (max_sample_row, max_sample_col)))

        self.file_count = len(filenames)
        print(f'Found {self.file_count} file(s)')

    def _source_transform(self, img: Image) -> torch.Tensor:
        """
        Applies transforms to the overall source image (not samples of the image).
        The original object is not modified.

        :param img: image to transform
        :return: transformed copy of image
        """
        img = T.PILToTensor()(img)
        return img

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        index = np.random.randint(0, self.file_count)
        source_img, (max_row, max_col) = self.imgs[index]
        start_row = np.random.randint(0, max_row)
        start_col = np.random.randint(0, max_col)
        sample = source_img[:, start_row:start_row + self.initial_sample_size, start_col:start_col + self.initial_sample_size]
        return sample
