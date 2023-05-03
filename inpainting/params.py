"""
Program parameters.
"""

from pathlib import Path

import torchvision.transforms as T


class Params:
    """
    Data class representing run parameters. All default values are valid.
    Parameters are generally NOT checked for validity.
    """
    data_dir_path: Path = None
    save_dir_path: Path = None
    experiment_name: str = None

    dataset_name: str = ''

    channel_count = 3
    initial_sample_size = 181  # ensures center crop on a 45° rotation is a full 128x128 with no gaps
    img_size = 128
    mask_size = 16
    samples_per_epoch = 20000
    wandering_mask: bool = False

    epoch_count = 25
    batch_size = 64
    lr = 1e-3
    lr_scheduler_step_freq = 500  # number of batches before scheduler possibly alters lr
    b1 = 0.5  # adam: decay of first order momentum of gradient
    b2 = 0.999  # adam: decay of first order momentum of gradient
    train_worker_count = 6  # must be at most number of CPUs
    augs_per_example = 1  # number of image augmentations per training example; minimum 1

    def get_transform(self):
        if self.dataset_name == 'dtd':
            return T.Compose([
                # T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                # T.RandomResizedCrop((self.img_size, self.img_size), scale=(0.5, 1)),
                # T.RandomRotation(45, expand=False),

                # T.Resize((self.img_size * 2, self.img_size * 2), antialias=None),
                # T.CenterCrop((self.img_size, self.img_size))
                T.Resize((self.img_size, self.img_size), antialias=True)
            ])
        elif self.dataset_name == 'celeb':
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),  # doubles dataset size
                T.Resize((self.img_size, self.img_size), antialias=True)
            ])
        elif self.dataset_name == 'bigtex':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(45, expand=True),
                T.Resize((self.img_size * 2, self.img_size * 2), antialias=None),
                T.CenterCrop((self.img_size, self.img_size)),
            ])
