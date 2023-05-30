"""
Project runner.
"""

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.masked_img_dataset import MaskedImageDataset
from src.params import Params
from src.train import InPaintingGAN
from src.util import *


class ImageSamplingProfiles:
    SUBSAMPLING = 'subsampling'
    SINGLE_SAMPLE = 'single_sample'

    @property
    def values(self):
        return [
            self.SUBSAMPLING,
            self.SINGLE_SAMPLE
        ]


def run(params: Params) -> None:
    """
    Run it!

    :param params: program parameters obj
    :return: None
    """
    print_torch_version_info()
    print(params)
    print('\n\tMonitor progress with Tensorboard, e.g. "tensorboard --logdir <save_dir_path>"\n')

    # TODO: implement training resumption
    # to resume training from saved state (easy): https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state

    logger = TensorBoardLogger(params.save_dir_path, name=params.experiment_name)
    trainer = L.Trainer(accelerator="gpu", max_epochs=params.epoch_count, logger=logger, log_every_n_steps=50, detect_anomaly=True)

    dataset = MaskedImageDataset(params.data_dir_path,
                                 params.samples_per_img,
                                 params.augs_per_sample,
                                 params.img_size,
                                 params.mask_size,
                                 params.wandering_mask_prob,
                                 params.base_transform,
                                 params.aug_transform)

    train_dataloader = DataLoader(dataset,
                                  batch_size=params.batch_size,
                                  num_workers=params.train_worker_count,
                                  persistent_workers=True
                                  )

    model = InPaintingGAN(dataset,
                          params.img_size,
                          params.mask_size,
                          params.b1,
                          params.b2,
                          params.lr,
                          params.lr_scheduler_step_freq)

    trainer.fit(model, train_dataloader)

    path = params.save_dir_path / 'model.pth'
    torch.save(model.state_dict(), path)

    print('\nTraining complete.')
    print(f'Model saved to "{path}"\n')
