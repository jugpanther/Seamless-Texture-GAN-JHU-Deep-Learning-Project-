"""
Celeb dataset specifics.
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from inpainting.dtd import MaskedImageDataset
from inpainting.params import Params


def get_celeb_dataloader(split_name: str, params: Params) -> DataLoader:
    """
    Builds a celeb dataloader.

    :param split_name: must be one of ['train', 'val', 'test']
    :param params: params obj
    :return: dataloader
    """
    assert split_name in ['train', 'val', 'test'], f'Unknown split name: {split_name}'
    celeb_data = CelebDataset(params.data_dir_path)
    dataset = MaskedImageDataset(celeb_data, params.img_size, params.mask_size, transform=params.get_transform(), augs_per_example=params.augs_per_example)

    dl = DataLoader(dataset,
                    batch_size=params.batch_size,
                    num_workers=params.train_worker_count,
                    persistent_workers=True
                    )
    return dl


class CelebDataset(Dataset):
    """
    Data class that loads Celeb256 images.
    """

    def __init__(self, img_dir_path: Path):
        self.data_path = img_dir_path
        self.filenames = []
        self.file_count = 0
        self._load_filenames()
        self.imgs = []
        self._data_preloaded = False
        # self._preload()  # preloads all images into memory

    def _load_filenames(self):
        print(f'Loading Celeb256 files from {self.data_path}')
        self.filenames = [name for name in os.listdir(self.data_path) if os.path.isfile(str(self.data_path / name))]
        self.file_count = len(self.filenames)
        print(f'Found {self.file_count} files.')

    def _preload(self):
        print('Preloading images...')
        for i in tqdm(range(self.file_count), leave=False):
            self.imgs.append(self.__getitem__(i))
        self._data_preloaded = True

    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        if self._data_preloaded:
            return self.imgs[idx]
        else:
            img = Image.open(self.data_path / self.filenames[idx]).copy()
            return img
