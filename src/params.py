"""
Program parameters.
"""

from pathlib import Path

import torchvision.transforms as T

from src.augmentation import AugTypes


class Params:
    """
    Data class representing run parameters.
    """
    data_dir_path: Path = None
    save_dir_path: Path = None
    experiment_name: str = None
    run_profile: str = None

    channel_count = -1
    img_size = -1
    mask_size = -1
    samples_per_img = -1
    wandering_mask_prob: float = 0
    test_texture_size: int = -1

    epoch_count = -1
    batch_size = -1
    lr = -1
    lr_scheduler_step_freq = 500  # number of batches before scheduler possibly alters lr
    b1 = 0.5  # adam: decay of first order momentum of gradient
    b2 = 0.999  # adam: decay of first order momentum of gradient
    train_worker_count = 6

    aug_type: int = None
    augs_per_sample = -1
    aug_transform: T.Compose = None
    base_transform: T.Compose = None

    @property
    def true_tile_size(self) -> int:
        """
        Calculates the "true" tile size from the input image size and mask border size.

        :return: true tile size
        """
        return self.img_size - 2 * self.mask_size

    def __str__(self):
        return f'Dataset location:   "{self.data_dir_path}"\n' \
               f'Save data location: "{self.save_dir_path}"\n' \
               f'Run profile is "{self.run_profile}"' \
               f'Generator input size is {self.img_size}² with a mask size of {self.mask_size}\n; true tile size is {self.true_tile_size}' \
               f'Taking {self.samples_per_img} samples per image with {self.augs_per_sample} augmentations per sample\n' \
               f'Wandering mask patch probability is {self.wandering_mask_prob}\n' \
               f'Test texture size is {self.test_texture_size}² (tiles), or {self.test_texture_size * self.true_tile_size}² (pixels)' \
               f'Training will run for {self.epoch_count} epochs with a batch size of {self.batch_size} and learning rate of {self.lr:.2e}\n' \
               f'Augmentation type is {AugTypes.as_str(self.aug_type)}\n' \
               f'Running with {self.train_worker_count} workers'
