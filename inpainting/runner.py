"""
Project runner.
"""

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from inpainting.bigtex import get_bigtex_dataloader
from inpainting.celeb import get_celeb_dataloader
from inpainting.dtd import get_dtd_dataloader
from inpainting.params import Params
from inpainting.train import InPaintingGAN
from inpainting.util import *


def run(params: Params) -> None:
    """
    Run it!

    :param params: program parameters obj
    :return: None
    """
    print_torch_version_info()

    print('\n\tMonitor progress with Tensorboard (must be started separately).')
    print('\te.g. "tensorboard --logdir <save_dir_path>"\n')

    # TODO: implement training resumption
    # to resume training from saved state (easy): https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state

    logger = TensorBoardLogger(params.save_dir_path, name=params.experiment_name)
    trainer = L.Trainer(accelerator="gpu", max_epochs=params.epoch_count, logger=logger, log_every_n_steps=50, detect_anomaly=True)

    if params.dataset_name == 'dtd':
        print('Using DTD dataset')
        train_dataloader = get_dtd_dataloader('train', params)
    elif params.dataset_name == 'celeb':
        print('Using Celeb256 dataset')
        train_dataloader = get_celeb_dataloader('train', params)
    elif params.dataset_name == 'bigtex':
        print('Using Big Textures dataset')
        train_dataloader = get_bigtex_dataloader('train', params)
    else:
        raise ValueError(f'Unknown dataset name "{params.dataset_name}" -- must be one of ["dtd", "celeb", "bigtex"]')

    model = InPaintingGAN(train_dataloader.dataset,
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
    print(f'Model saved to "{path}"')
