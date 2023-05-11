"""
Project-specific dataset class that supports image masking and augmentation.
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset

from src.util import channelwise_normalize_


class MaskedImageDataset(Dataset):
    """
    A Dataset class that supports image masking and augmentation.
    """

    def __init__(self,
                 img_dir_path: Path,
                 samples_per_img: int,
                 augs_per_sample: int,
                 img_size: int,
                 mask_size: int,
                 wandering_mask_prob: float = 0,
                 base_transform: T.Compose | None = None,
                 transform: T.Compose | None = None):
        """
        :param img_dir_path: path to directory containing all images
        :param samples_per_img: number of samples taken per image; must be 1 or more
        :param augs_per_sample: number of augmentations per sample; must be 0 or more
        :param img_size: image size, e.g. 64 for 64x64
        :param mask_size: border size (thickness) of mask edges in pixels, e.g. 8
        :param wandering_mask_prob: probability of applying the wandering mask patch to a given sample
        :param base_transform: optional; the fundamental transform without augmentation, such as a crop or resize
        :param transform: optional; random augmentation transform
        """
        super().__init__()
        self.data_path = img_dir_path
        self.samples_per_img = max(1, samples_per_img)
        self.augs_per_sample = max(0, augs_per_sample)  # zero indicates no augmentation at all
        self.img_size = img_size
        self.mask_size = mask_size
        self.img_channel_count = 3

        self.imgs = []
        self.file_count = 0
        self.filenames = []
        self._load_filenames()
        self.pil_to_tensor = T.PILToTensor()
        self._data_preloaded = False
        self.total_dataset_size = self.file_count * self.samples_per_img * (self.augs_per_sample + 1)

        self._corner_masks = []
        self._edge_masks = []
        self._prepare_mask_components()
        self.wandering_mask_prob = wandering_mask_prob

        self.base_transform = base_transform if base_transform is not None else T.Compose([])
        self.transform = transform if transform is not None else T.Compose([])

    def _load_filenames(self) -> None:
        """
        Collects all filenames in the specified directory. Does not load files.

        :return: None
        """
        print(f'Loading files from {self.data_path}...')
        self.filenames = [name for name in os.listdir(self.data_path) if os.path.isfile(str(self.data_path / name))]
        self.file_count = len(self.filenames)
        print(f'Found {self.file_count} files')

    def _prepare_mask_components(self):
        """
        Precomputes image mask components for faster mask generation.

        :return: None
        """
        corner_masks = []
        edge_masks = []

        corner_mask = np.zeros((self.img_size, self.img_size, self.img_channel_count))
        corner_mask[0:self.mask_size, 0:self.mask_size, :] = 1
        corner_masks.append(corner_mask.copy())

        edge_mask = np.zeros_like(corner_mask)
        edge_mask[0:self.mask_size, self.mask_size:self.img_size - self.mask_size, :] = 1
        edge_masks.append(edge_mask.copy())

        for _ in range(3):
            corner_mask = np.rot90(corner_mask)
            corner_masks.append(corner_mask)
            edge_mask = np.rot90(edge_mask)
            edge_masks.append(edge_mask)

        self._corner_masks = corner_masks
        self._edge_masks = edge_masks

    def create_mask(self) -> Tuple[Tensor, Tensor]:
        """
        Randomly creates a new mask.

        :return: mask tensor
        """
        corner_count = np.random.randint(0, 4 + 1)  # possibly zero
        corner_indexes = np.random.choice([0, 1, 2, 3], corner_count, replace=False)
        edge_count = np.random.randint(1, 3 + 1)  # minimum 1 (always have a minimum of one seed region)
        edge_indexes = np.random.choice([0, 1, 2, 3], edge_count, replace=False)

        mask = np.zeros((self.img_size, self.img_size, self.img_channel_count))

        for i in corner_indexes:
            mask = mask + self._corner_masks[i]

        for i in edge_indexes:
            mask = mask + self._edge_masks[i]

        # sometimes, add random "wandering mask patch" to ensure center of model is exposed to ground truth data
        if np.random.rand() < self.wandering_mask_prob:
            hole_size = self.img_size // 3
            start_row = np.random.randint(0, self.img_size - hole_size)
            start_col = np.random.randint(0, self.img_size - hole_size)
            mask[start_row:start_row + hole_size, start_col:start_col + hole_size] = 1

        mask = torch.asarray(mask, dtype=torch.float32)
        mask = torch.permute(mask, (2, 0, 1))

        # create noise-filled mask
        noise_mask = torch.rand_like(mask)
        noise_mask = noise_mask * (1 - mask)

        return mask, noise_mask

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx
            if step != 1 or stop < start:
                raise NotImplementedError('Nontrivial slicing is not implemented')
            result = [None] * (stop - start)
            for i, index in enumerate(range(start, stop)):
                result[i] = self._get_at(index)
            return result
        else:
            return self._get_at(idx)

    def _get_at(self, idx) -> Tuple:
        """
        Retrieves a single data sample at the given index.

        :param idx: index
        :return: tuple of (original image, logical mask, normalized masked image)
        """
        img_idx = idx % self.file_count
        source_img = self.imgs[img_idx]
        aug_idx = idx % self.augs_per_sample

        if aug_idx != 0:
            sample = self.base_transform(source_img)
        else:
            sample = self.transform(source_img)

        sample = self.pil_to_tensor(sample)

        sample = sample / 255.0
        assert sample.dtype.is_floating_point, f'Initial image value scaling failed to produce floats in range [0,1]; tensor dtype is {sample.dtype}'
        sample = self.transform(sample)
        logical_mask, noise_mask = self.create_mask()

        # create masked image
        masked_img = sample * logical_mask + noise_mask
        channelwise_normalize_(masked_img, 0, 0.5)

        return sample, logical_mask, masked_img

# def test_mask_generation():
#     img_size = 64
#
#     class MinimalDataset(Dataset):
#         def __getitem__(self, item):
#             return torch.randint(0, 255, (img_size, img_size, 3))
#
#     rows = 8
#     cols = 8
#     dataset = MaskedImageDataset(MinimalDataset(), img_size, 8)
#     mask_img_list = [dataset.create_mask() for _ in range(rows * cols)]
#     grid = make_grid(mask_img_list, rows, pad_value=0.5)
#     grid = F.to_pil_image(grid)
#     cv2.imshow('Test masks', np.asarray(grid))
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     test_mask_generation()
