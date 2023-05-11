from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm


class BaseDataset(Dataset):

    def __init__(self,
                 img_dir_path: Path,
                 base_transform: T.Compose | None,
                 transform: T.Compose | None,
                 samples_per_img: int,
                 augs_per_sample: int):
        """
        :param img_dir_path: path to directory containing all images
        :param base_transform: optional; the fundamental transform without augmentation, such as a crop or resize
        :param transform: optional; random augmentation transform
        :param samples_per_img: number of samples taken per image; must be 1 or more
        :param augs_per_sample: number of augmentations per sample; must be 0 or more
        """
        self.data_path = img_dir_path
        self.base_transform = base_transform
        self.transform = transform if transform is not None else T.Compose([])
        self.samples_per_img = max(1, samples_per_img)
        self.augs_per_sample = max(0, augs_per_sample)  # zero indicates no augmentation at all

        self.imgs = []
        self.file_count = 0
        self.filenames = []
        self._load_filenames()
        self.pil_to_tensor = T.PILToTensor()
        self._data_preloaded = False
        self.total_dataset_size = self.file_count * self.samples_per_img * (self.augs_per_sample + 1)

    def _preload(self) -> None:
        """
        Preloads all specified files into memory.

        :return: None
        """
        print('Preloading images...')
        for i in tqdm(range(self.file_count), leave=False):
            self.imgs.append(self.__getitem__(i))
        self._data_preloaded = True

    def _load_filenames(self) -> None:
        """
        Collects all filenames to use during training. Does not load files.

        :return: None
        """
        raise NotImplementedError()  # override in derived classes

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, idx):
        img_idx = idx % self.file_count
        source_img = self.imgs[img_idx]
        aug_idx = idx % self.augs_per_sample

        if aug_idx != 0:
            sample = self.base_transform(source_img)
        else:
            sample = self.transform(source_img)

        return sample
