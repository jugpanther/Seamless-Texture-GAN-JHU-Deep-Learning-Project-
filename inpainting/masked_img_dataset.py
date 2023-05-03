"""
Project-specific dataset class that supports image masking and augmentation.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from inpainting.util import channelwise_normalize_


class MaskedImageDataset(Dataset):
    """
    A Dataset class that supports image masking and augmentation.
    BACKING DATASET MUST PROVIDE IMAGES IN [0,255] RANGE!
    """

    forbidden_user_transforms = (T.PILToTensor, T.Normalize)

    def __init__(self,
                 backing_dataset: Dataset,
                 img_size: int,
                 mask_size: int,
                 img_channels: int = 3,
                 transform: T.Compose | None = None,
                 augs_per_example: int = 1,
                 use_wandering_mask: bool = False):
        """
        :param backing_dataset: ordinary Dataset instance that provides square images
        :param img_size: image size, e.g. 64 for 64x64
        :param mask_size: border size (thickness) of mask edges in pixels, e.g. 8
        :param img_channels: number of channels in each image
        :param transform: (optional) user-supplied augmentations. Do not include any forbidden transforms (see class property).
        :param augs_per_example: number of augmentations to generate per single example. Must be at least 1.
        """
        super().__init__()
        self.data = backing_dataset
        if self.data[0].max().item() <= 1:
            import warnings
            warnings.warn(f'{self.__name__} expects the backing dataset to provide image data in range [0,255]. The image at index 0 does not seem to follow this convention. This check is NOT '
                          f'performed on all images.')

        self.img_size = img_size
        self.mask_size = mask_size
        self.img_channel_count = img_channels
        self.transform = transform if transform is not None else T.Compose([])
        self.augs_per_example = max(1, augs_per_example)
        self._enforce_valid_transforms()

        self.corner_masks = []
        self.edge_masks = []
        self._prepare_mask_parts()
        self.use_wandering_mask = use_wandering_mask

    def _enforce_valid_transforms(self) -> None:
        """
        Raises an exception if user-supplied transforms are not allowed.
        (Some transforms are performed internally at specific points under specific conditions.)
        :return: None
        """
        for transform in self.transform.transforms:
            if isinstance(transform, self.forbidden_user_transforms):
                raise ValueError(f'Please remove any transforms that match these: {self.forbidden_user_transforms}. These are handled automatically.')

    def _prepare_mask_parts(self):
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

        self.corner_masks = corner_masks
        self.edge_masks = edge_masks

    def create_mask(self) -> Tuple[Tensor, Tensor]:
        """
        Randomly creates a new mask shape.
        :return: mask tensor
        """
        corner_count = np.random.randint(0, 4 + 1)  # possibly zero
        corner_indexes = np.random.choice([0, 1, 2, 3], corner_count, replace=False)
        edge_count = np.random.randint(1, 3 + 1)  # minimum 1 (always have a minimum of one seed region)
        edge_indexes = np.random.choice([0, 1, 2, 3], edge_count, replace=False)

        mask = np.zeros((self.img_size, self.img_size, self.img_channel_count))

        for i in corner_indexes:
            mask = mask + self.corner_masks[i]

        for i in edge_indexes:
            mask = mask + self.edge_masks[i]

        # sometimes, add random "wandering hole" to ensure center of model learns too
        if self.use_wandering_mask and np.random.rand() > 0.5:
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
        return len(self.data) * self.augs_per_example

    def __getitem__(self, idx):
        """
        Ordinary __getitem__ behavior. Supports slicing or single indexing.
        Model requirements for reference:
        =================================
        Generator:
            input: (normalized) masked image
            output: true color mask patch
        Discriminator:
            input: true color image (full original image or original and generated true color image)
        """
        if isinstance(idx, slice):
            start, stop, step = idx
            if step != 1 or stop < start:
                raise NotImplementedError('Complex slicing is not implemented')
            result = [None] * (stop - start)
            for i, index in enumerate(range(start, stop)):
                result[i] = self._get_at(index)
            return result
        else:
            return self._get_at(idx)

    def _get_at(self, idx) -> Tuple:
        """
        Retrieves a single data item at the given index.

        :param idx: index
        :return: tuple of (original image, logical mask, normalized masked image)
        """
        img_orig = self.data[idx // self.augs_per_example]

        if not isinstance(img_orig, Tensor):
            raise ValueError(f'Unexpected source image type: {type(img_orig)}; must be torch.Tensor. '
                             f'The __getitem__ function of your dataset class must return a tensor.')

        # do not resize the original image here. If that is necessary, it should be part of the user-provided transform.
        img_orig = img_orig / 255.0
        assert img_orig.dtype.is_floating_point, f'Initial image value scaling failed to produce floats in range [0,1]; tensor dtype is {img_orig.dtype}'
        img_orig = self.transform(img_orig)
        logical_mask, noise_mask = self.create_mask()

        # create masked image
        masked_img = img_orig * logical_mask + noise_mask
        channelwise_normalize_(masked_img, 0, 0.5)

        return img_orig, logical_mask, masked_img


def test_mask_generation():
    img_size = 64

    class MinimalDataset(Dataset):
        def __getitem__(self, item):
            return torch.randint(0, 255, (img_size, img_size, 3))

    rows = 8
    cols = 8
    dataset = MaskedImageDataset(MinimalDataset(), img_size, 8)
    mask_img_list = [dataset.create_mask() for _ in range(rows * cols)]
    grid = make_grid(mask_img_list, rows, pad_value=0.5)
    grid = F.to_pil_image(grid)
    cv2.imshow('Test masks', np.asarray(grid))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_mask_generation()
