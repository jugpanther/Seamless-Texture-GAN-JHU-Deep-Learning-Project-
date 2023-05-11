"""
DTD dataset specifics.
"""

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD

from src.masked_img_dataset import MaskedImageDataset
from src.params import Params


# https://pytorch.org/vision/main/generated/torchvision.datasets.DTD.html


def get_dtd_dataloader(split_name, params: Params) -> DataLoader:
    """
    Builds a DTD dataloader.

    :param split_name: must be one of ['train', 'val', 'test']
    :param params: params obj
    :return: dataloader
    """
    assert split_name in ['train', 'val', 'test'], f'Unknown DTD split name: {split_name}'
    dtd_data = DTD(root=str(params.data_dir_path), split=split_name, transform=None, target_transform=None, download=True)
    wrapped_dtd = DTDWrapper(dtd_data)
    dataset = MaskedImageDataset(wrapped_dtd, params.img_size, params.mask_size, transform=params.get_transform(), augs_per_example=params.augs_per_example)

    dl = DataLoader(dataset,
                    batch_size=params.batch_size,
                    num_workers=params.train_worker_count,
                    persistent_workers=True
                    )
    return dl


class DTDWrapper(Dataset):
    """
    Data class that wraps PyTorch's DTD dataset to remove labels.
    """

    def __init__(self, dtd_data: DTD):
        self.data = dtd_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0]
