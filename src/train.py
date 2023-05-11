"""
Model training code.
"""

from typing import List, Tuple
import cv2
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm
from skimage.metrics import structural_similarity

from src.discriminator import Discriminator
from src.generator import Generator
from src.masked_img_dataset import MaskedImageDataset
from src.multi_res_generator import MultiResGenerator
from src.results_grid import ResultsGrid
from src.texturing import TextureBuilder
from src.util import model_sanity_check, tensor01_to_RGB01


class InPaintingGAN(L.LightningModule):

    def __init__(self, dataset: MaskedImageDataset, img_size, mask_size, adam_b1, adam_b2, lr, lr_sched_step_freq, activation_fn):
        """
        Creates a new Lightning model for RGB image inpainting.

        :param dataset: underlying dataset
        :param img_size: image size
        :param adam_b1: Adam optimizer parameter
        :param adam_b2: Adam optimizer parameter
        :param lr: learning rate
        :param lr_sched_step_freq: steps between lr scheduler inspection
        """
        super().__init__()
        self.dataset = dataset
        self.img_size = img_size
        self.mask_size = mask_size
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.lr = lr
        self.lr_sched_step_freq = lr_sched_step_freq

        # self.generator = Generator(activation_fn=activation_fn)
        # self.discriminator = Discriminator()

        self.generator = MultiResGenerator()
        self.discriminator = Discriminator()

        self.verify_models()
        self.automatic_optimization = False  # because we have multiple optimizers

        self.fraction_complete = 0

    def verify_models(self) -> None:
        """
        Validates the generator and discriminator models. Raises an exception if either fails.

        :return: None
        """
        model_sanity_check(self.generator, (3, self.img_size, self.img_size), (3, self.img_size, self.img_size), 'Generator')
        model_sanity_check(self.discriminator, (3, self.img_size, self.img_size), (1, self.img_size, self.img_size), 'Discriminator')
        print('Model passed data size checks')

    def forward(self, x):
        return self.generator(x)

    def on_train_start(self) -> None:
        self.epoch_tqdm = tqdm(total=self.trainer.max_epochs - 1, leave=True)

    def update_epoch_tqdm(self):
        if self.epoch_tqdm is not None:
            self.epoch_tqdm.desc = f'Epoch {self.current_epoch}/{self.trainer.max_epochs - 1}'
            if self.current_epoch > 0:
                self.epoch_tqdm.update(1)

    def on_train_epoch_start(self) -> None:
        self.fraction_complete = self.current_epoch / self.trainer.max_epochs
        self.update_epoch_tqdm()

    def adversarial_loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def pixelwise_loss(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)

    def build_discriminator_ground_truths(self, original_imgs: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Constructs the ground truth for the discriminator for a specific batch.
        The ground truth consists of two single-channel 2D images:
            • a "fully true" tensor to be used with original images
            • a "partially true" (fake) tensor to be used with generator outputs

        :param original_imgs: tensor of unmodified images
        :param masks: tensor of masks
        :return: tuple of (ground truth real, ground truth fake)
        """
        batch_size = original_imgs.shape[0]

        # value of 1 represents real, value of 0 represents fake
        # a "real" image will be fully real (value of 1 everywhere)
        # a "fake" image will have a variety of 0s and 1s depending on each mask in the batch

        ideal_d_output_real = torch.ones(batch_size, 1, self.img_size, self.img_size).requires_grad_(False)
        ideal_d_output_fake = masks[:, :1].clone().detach().requires_grad_(False)  # take first channel of mask only because this tensor needs to be 1D

        # move tensors to same devices as other batch data
        ideal_d_output_real = ideal_d_output_real.type_as(original_imgs)
        ideal_d_output_fake = ideal_d_output_fake.type_as(original_imgs)

        return ideal_d_output_real, ideal_d_output_fake

    def training_step(self, batch, batch_idx):
        """
        Runs a single training iteration with a single batch of data.

        :param batch: batch data; combined output of dataset __getitem__ called many times (possibly output of a CollateFn)
        :param batch_idx: batch index
        :return: None
        """
        # inspired from from https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
        imgs_orig, masks, masked_imgs = batch  # type: Tensor, Tensor, Tensor

        # # pre-train generator on pure noise to have a better chance at understanding noise in the masked images
        # if self.current_epoch < 10:
        #     mixer = torch.round(torch.rand_like(masked_imgs))
        #     masked_imgs = imgs_orig * mixer + torch.rand_like(masked_imgs) * (1 - mixer)
        #     masks = mixer

        self.generator.train()  # possibly not needed, but explicit
        self.discriminator.train()

        optimizer_G, optimizer_D = self.optimizers()  # type: torch.optim.Optimizer
        lr_sch_G, lr_sch_D = self.lr_schedulers()  # type: torch.optim.lr_scheduler.ReduceLROnPlateau

        # get ground truths
        ideal_D_output_real, ideal_D_output_fake = self.build_discriminator_ground_truths(imgs_orig, masks)

        # ===================================================================================================================
        # train the generator
        self.toggle_optimizer(optimizer_G)
        G_output = self.generator(masked_imgs)

        # painted_imgs.requires_grad_(False)
        G_output_for_D = torch.zeros_like(G_output)
        G_output_for_D.copy_(G_output)
        G_output_for_D.requires_grad_(True)

        D_output = self.discriminator(G_output)  # see what D thinks of G's attempt
        G_adv_loss = self.adversarial_loss(D_output, ideal_D_output_real.detach().requires_grad_(True))  # ideally zero; trying to make D guess all images are real
        G_pixel_loss = self.pixelwise_loss(G_output, imgs_orig.detach().requires_grad_(True))  # ideally zero; trying to make G produce exact, real patches

        # adv_loss_weight = (0.2 - 0.01) * np.exp(-3 * self.fraction_complete) + 0.01  # https://www.desmos.com/calculator/elzcvwukqw
        adv_loss_weight = 0.1
        # pixel_loss_weight = 0.8 + self.fraction_complete * (0.99 - 0.8)
        pixel_loss_weight = 0.9
        # total_G_loss = (0.001 * G_adv_loss) + (0.999 * G_pixel_loss)  # ideally zero
        total_G_loss = (adv_loss_weight * G_adv_loss) + (pixel_loss_weight * G_pixel_loss)  # ideally zero
        # total_G_loss = (0.1 * G_adv_loss) + (0.9 * G_pixel_loss)  # ideally zero
        # total_G_loss = (0.5 * G_adv_loss) + (0.5 * G_pixel_loss)  # ideally zero

        # following this for order of operations: https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html#use-multiple-optimizers-like-gans
        optimizer_G.zero_grad()
        self.manual_backward(total_G_loss, retain_graph=True)
        optimizer_G.step()
        # lr_sch_G.step(total_G_loss)
        self.untoggle_optimizer(optimizer_G)

        # ===================================================================================================================
        # train discriminator
        self.toggle_optimizer(optimizer_D)

        # assess ability to correctly identify real images
        D_output_real = self.discriminator(imgs_orig.requires_grad_(True))
        D_loss_real = self.adversarial_loss(D_output_real, ideal_D_output_real.requires_grad_(True))

        # assess ability to correctly identify fake images
        D_output_fake = self.discriminator(G_output_for_D)  # modified masked images from generator
        D_loss_fake = self.adversarial_loss(D_output_fake, ideal_D_output_fake.requires_grad_(True))
        total_D_loss = (D_loss_real + D_loss_fake) / 2.0  # average loss

        optimizer_D.zero_grad()
        self.manual_backward(total_D_loss)
        optimizer_D.step()
        # lr_sch_D.step(total_D_loss)
        self.untoggle_optimizer(optimizer_D)

        # ===================================================================================================================
        self.log('g_loss', total_G_loss, prog_bar=True, logger=True, on_step=True)
        self.log('d_loss', total_D_loss, prog_bar=True, logger=True, on_step=True)
        self.log('G_adv_loss', G_adv_loss, prog_bar=False, logger=True, on_step=True)
        self.log('G_pixel_loss', G_pixel_loss, prog_bar=False, logger=True, on_step=True)
        self.log('D_loss_real', D_loss_real, prog_bar=False, logger=True, on_step=True)
        self.log('D_loss_fake', D_loss_fake, prog_bar=False, logger=True, on_step=True)
        # self.log('g_lr', lr_sch_G._last_lr[0], prog_bar=False, logger=True, on_step=True)
        # self.log('d_lr', lr_sch_D._last_lr[0], prog_bar=False, logger=True, on_step=True)

        # log extra data a few times per epoch
        if batch_idx % (self.trainer.num_training_batches // 5) == 0:
            self.log_results_grid(batch_idx)
            self.log_texture_demo(batch_idx)

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.adam_b1, self.adam_b2))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.adam_b1, self.adam_b2))
        lr_sch_config_G = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.25, patience=1000, min_lr=1e-6, cooldown=1000),
            "interval": "step",
            "frequency": self.lr_sched_step_freq,
            "name": None,
        }
        lr_sch_config_D = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_D, factor=0.25, patience=1000, min_lr=1e-6, cooldown=1000),
            "interval": "step",
            "frequency": self.lr_sched_step_freq,
            "name": None,
        }
        return [opt_G, opt_D], [lr_sch_config_G, lr_sch_config_D]

    def log_results_grid(self, batch_idx) -> None:
        """
        Sends a results image to the logger.

        :param batch_idx: current batch index
        :return: None
        """
        grid_img = self.build_results_grid()
        elapsed_steps = self.current_epoch * len(self.dataset) + batch_idx
        logger = self.logger  # type: TensorBoardLogger
        logger.experiment.add_image("generated_images", grid_img, elapsed_steps, dataformats='HWC')

    def build_results_grid(self) -> np.ndarray:
        """
        Constructs a results grid image.

        :return: image as RGB array in format [H,W,3]
        """
        pnsr_sum = 0
        ssim_sum = 0

        with torch.no_grad():
            self.discriminator.eval()
            sample_count = 6
            dataset = self.dataset

            # columns are original image, masked image, generator output, discriminator output
            grid = ResultsGrid(rows=sample_count, cols=4, tile_size=128)

            for i in range(sample_count):
                i = len(dataset) // sample_count * i
                data = dataset[i]

                # col 0: original image 
                img_orig = data[0]
                grid.add_tile(img_orig)
                metric_img_real = img_orig.numpy().transpose(1, 2, 0)

                # col 1: masked image
                mask = data[1]
                pretty_mask = img_orig * mask
                grid.add_tile(pretty_mask)

                # col 2: generator output
                masked_img = data[2]
                G_output = self.forward(masked_img.unsqueeze(0).to(self.device))
                grid.add_tile(G_output.detach())
                metric_img_fake = G_output[0].detach().cpu().permute(1, 2, 0).numpy()

                # col 3: discriminator output
                D_output = self.discriminator(G_output)
                grid.add_tile(D_output, normalize=False)

                # metrics
                pnsr_sum += cv2.PSNR(metric_img_real, metric_img_fake)
                mean_ssim, full_ssim = structural_similarity(metric_img_real, metric_img_fake, channel_axis=2, gaussian_weights=True, use_sample_covariance=False, full=True)
                ssim_sum += mean_ssim

            grid_img = grid.get().copy()
            del grid, G_output, D_output

            self.log('PNSR', pnsr_sum / sample_count, prog_bar=False, logger=True, on_step=True)
            self.log('SSIM', ssim_sum / sample_count, prog_bar=False, logger=True, on_step=True)

            return grid_img

    def log_texture_demo(self, batch_idx) -> None:
        """
        Sends a demonstration texture to the logger.

        :param batch_idx: current batch index
        :return: None
        """
        rows = 5
        cols = 5
        seed_img = tensor01_to_RGB01(self.dataset[0][0])
        builder = TextureBuilder(self.img_size, rows, cols, self.generator, self.mask_size, self.device, seed_img)
        builder.build()
        texture_img = builder.get()
        elapsed_steps = self.current_epoch * len(self.dataset) + batch_idx
        logger = self.logger  # type: TensorBoardLogger
        logger.experiment.add_image("textures", texture_img, elapsed_steps, dataformats='HWC')

        builder = TextureBuilder(self.img_size, rows, cols, self.generator, self.mask_size * 2, self.device, seed_img)
        builder.build()
        texture_img = builder.get()
        elapsed_steps = self.current_epoch * len(self.dataset) + batch_idx
        logger = self.logger  # type: TensorBoardLogger
        logger.experiment.add_image("textures*2", texture_img, elapsed_steps, dataformats='HWC')
