"""
Model training code.
"""

from typing import Tuple

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from skimage.metrics import structural_similarity
from torch import Tensor
from tqdm import tqdm

from src.masked_img_dataset import MaskedImageDataset
from src.multi_res_generator import MultiResGenerator
from src.results_grid import ResultsGrid
from src.texture_discriminator import TextureDiscriminator
from src.texturing import TextureBuilder
from src.tile_discriminator import TileDiscriminator
from src.util import model_sanity_check, tensor01_to_RGB01


class InPaintingGAN(L.LightningModule):

    def __init__(self, dataset: MaskedImageDataset, img_size, mask_size, texture_builder: TextureBuilder, adam_b1, adam_b2, lr, lr_sched_step_freq):
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
        self.texture_builder = texture_builder
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.lr = lr
        self.lr_sched_step_freq = lr_sched_step_freq

        self.generator = MultiResGenerator()
        self.tile_discriminator = TileDiscriminator()
        self.texture_discriminator = TextureDiscriminator()

        self.verify_models()
        self.automatic_optimization = False  # because we have multiple optimizers

        self.epoch_tqdm = None

    def verify_models(self) -> None:
        """
        Validates the output sizes of each model. Raises an exception upon failure.

        :return: None
        """
        model_sanity_check(self.generator, (3, self.img_size, self.img_size), (3, self.img_size, self.img_size), 'Generator')
        model_sanity_check(self.tile_discriminator, (3, self.img_size, self.img_size), (1, self.img_size, self.img_size), 'Tile Discriminator')
        model_sanity_check(self.texture_discriminator, (3, self.texture_builder.texture_size, self.texture_builder.texture_size), (1,), 'Texture Discriminator')
        print('Models passed data size checks')

    def forward(self, x) -> Tensor:
        """
        Evaluates the generator using the given input.

        :param x: tensor input including batch dimension
        :return: tensor output
        """
        return self.generator(x)

    def on_train_start(self) -> None:
        """
        Built-in override. Configures a progress bar to track overall training progress.

        :return: None
        """
        self.epoch_tqdm = tqdm(total=self.trainer.max_epochs - 1, leave=True)

    def update_epoch_tqdm(self) -> None:
        """
        Increments the overall training progress bar.

        :return: None
        """
        if self.epoch_tqdm is not None:
            self.epoch_tqdm.desc = f'Epoch {self.current_epoch}/{self.trainer.max_epochs - 1}'
            if self.current_epoch > 0:
                self.epoch_tqdm.update(1)

    def on_train_epoch_start(self) -> None:
        """
        Built-in override. Called at the start of every training epoch.

        :return:
        """
        self.update_epoch_tqdm()

    @staticmethod
    def adversarial_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the adversarial loss between a prediction and ground truth.

        :param y_pred: predicted tensor
        :param y_true: ground truth tensor
        :return: adversarial loss tensor
        """
        return F.mse_loss(y_pred, y_true)

    @staticmethod
    def pixelwise_loss(y_pred, y_true) -> Tensor:
        """
        Computes the pixel-wise loss (pixel-wise similarity) between a prediction and ground truth.

        :param y_pred: predicted tensor
        :param y_true: ground truth tensor
        :return: pixel-wise loss tensor
        """
        return F.l1_loss(y_pred, y_true)

    @staticmethod
    def texture_loss(y_pred, y_true) -> Tensor:
        """
        Computes the loss of the predicted "real-ness" of a texture given the ground truth.

        :param y_pred: predicted value
        :param y_true: ground truth value
        :return: loss value
        """
        return F.binary_cross_entropy(y_pred, y_true).item()

    def build_tile_discriminator_ground_truths(self, original_imgs: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Constructs the ground truth for the tile discriminator for a specific batch.
        The ground truth consists of two single-channel 2D images:
            • a "fully true" tensor to be used with original images
            • a "partially true" (fake) tensor to be used with generator outputs

        :param original_imgs: tensor of unmodified images
        :param masks: tensor of masks
        :return: tuple of (ground truth real, ground truth fake)
        """
        batch_size = original_imgs.shape[0]

        # value of 1 represents real, value of 0 represents fake
        # --> a "real" image will be fully real (value of 1 everywhere)
        # --> a "fake" image will have a variety of 0s and 1s depending on each mask in the batch

        ideal_discr_output_real = torch.ones(batch_size, 1, self.img_size, self.img_size)
        ideal_discr_output_fake = masks[:, 0].clone()  # take first channel of mask only because this tensor needs to be single-channel

        # move tensors to same devices as other batch data
        ideal_discr_output_real = ideal_discr_output_real.type_as(original_imgs)
        ideal_discr_output_fake = ideal_discr_output_fake.type_as(original_imgs)

        return ideal_discr_output_real, ideal_discr_output_fake

    def train_generator(self, imgs_orig: Tensor, masked_imgs: Tensor, ideal_discr_output_real: Tensor, generated_texture: Tensor) -> Tensor:
        """
        Performs a training iteration on the generator.

        :param imgs_orig: batch data; original images
        :param masked_imgs: batch data; masked images
        :param ideal_discr_output_real: constructed ideal tile discriminator output
        :param generated_texture: assembled (fake) texture using generated tiles
        :return: raw generator output
        """
        optimizer_gen = self.optimizers()[0]
        self.toggle_optimizer(optimizer_gen)

        gen_output = self.generator(masked_imgs)
        tile_discr_output = self.tile_discriminator(gen_output)
        texture_discr_output = self.texture_discriminator(generated_texture)
        gen_adv_loss = self.adversarial_loss(tile_discr_output, ideal_discr_output_real)  # ideally zero; trying to make tile discriminator guess all images are real
        gen_pixel_loss = self.pixelwise_loss(gen_output, imgs_orig)  # ideally zero; trying to make generator produce exact, real tiles
        gen_texture_loss = 1 - texture_discr_output  # ideally zero; trying to make texture discriminator guess texture is real

        adv_loss_weight = 0.1
        pixel_loss_weight = 0.7
        texture_loss_weight = 0.2
        total_gen_loss = (adv_loss_weight * gen_adv_loss) + (pixel_loss_weight * gen_pixel_loss) + (texture_loss_weight * gen_texture_loss)  # ideally zero

        optimizer_gen.zero_grad()
        self.manual_backward(total_gen_loss, retain_graph=False)
        optimizer_gen.step()
        # lr_sch_gen = self.lr_schedulers()[0]
        # lr_sch_gen.step(total_gen_loss)
        self.untoggle_optimizer(optimizer_gen)

        self.log('g_loss', total_gen_loss, prog_bar=True, logger=True, on_step=True)
        self.log('g_adv_loss', gen_adv_loss, prog_bar=False, logger=True, on_step=True)
        self.log('g_pixel_loss', gen_pixel_loss, prog_bar=False, logger=True, on_step=True)
        # self.log('g_lr', lr_sch_gen._last_lr[0], prog_bar=False, logger=True, on_step=True)

        return gen_output

    def train_tile_discriminator(self, imgs_orig: Tensor, ideal_tile_discr_output_real: Tensor, ideal_tile_discr_output_fake: Tensor, gen_output: Tensor) -> None:
        """
        Performs a training iteration on the tile discriminator.

        :param imgs_orig: batch data; original images
        :param ideal_tile_discr_output_real: constructed ideal tile discriminator output for real examples
        :param ideal_tile_discr_output_fake: constructed ideal tile discriminator output for fake examples
        :param gen_output: example generator output to use for training on fake imagery
        :return: None
        """
        optimizer_tile_discr = self.optimizers()[1]
        self.toggle_optimizer(optimizer_tile_discr)

        # assess ability to correctly identify real images
        discr_output_real = self.tile_discriminator(imgs_orig)
        discr_loss_real = self.adversarial_loss(discr_output_real, ideal_tile_discr_output_real)

        # assess ability to correctly identify fake images
        discr_output_fake = self.tile_discriminator(gen_output)
        discr_loss_fake = self.adversarial_loss(discr_output_fake, ideal_tile_discr_output_fake)

        total_discr_loss = (discr_loss_real + discr_loss_fake) / 2.0  # average loss

        optimizer_tile_discr.zero_grad()
        self.manual_backward(total_discr_loss)
        optimizer_tile_discr.step()
        # lr_sch_tile_discr = self.lr_schedulers()[1]
        # lr_sch_tile_discr.step(total_discr_loss)
        self.untoggle_optimizer(optimizer_tile_discr)

        self.log('tile_d_loss', total_discr_loss, prog_bar=True, logger=True, on_step=True)
        self.log('tile_d_loss_real', discr_loss_real, prog_bar=False, logger=True, on_step=True)
        self.log('tile_d_loss_fake', discr_loss_fake, prog_bar=False, logger=True, on_step=True)
        # self.log('tile_d_lr', lr_sch_tile_discr._last_lr[0], prog_bar=False, logger=True, on_step=True)

    def train_texture_discriminator(self, generated_texture: Tensor, real_texture: Tensor) -> None:
        """
        Performs a training iteration on the texture discriminator.

        :param generated_texture: texture assembled from generated tiles (fake texture)
        :param real_texture: ground truth real texture from source
        :return: None
        """
        optimizer_texture_discr = self.optimizers()[2]
        self.toggle_optimizer(optimizer_texture_discr)

        # assess ability to correctly identify real textures
        discr_output_real = self.tile_discriminator(real_texture)  # ideally outputs 1
        discr_loss_real = self.texture_loss(Tensor([discr_output_real]), Tensor([1]))

        # assess ability to correctly identify fake images
        discr_output_fake = self.tile_discriminator(generated_texture)  # ideally outputs 0
        discr_loss_fake = self.texture_loss(Tensor([discr_output_fake]), Tensor([0]))

        total_discr_loss = (discr_loss_real + discr_loss_fake) / 2.0  # average loss

        optimizer_texture_discr.zero_grad()
        self.manual_backward(total_discr_loss)
        optimizer_texture_discr.step()
        # lr_sch_texture_discr = self.lr_schedulers()[2]
        # lr_sch_texture_discr.step(total_discr_loss)
        self.untoggle_optimizer(optimizer_texture_discr)

        self.log('texture_d_loss', total_discr_loss, prog_bar=True, logger=True, on_step=True)
        self.log('texture_d_loss_real', discr_loss_real, prog_bar=False, logger=True, on_step=True)
        self.log('texture_d_loss_fake', discr_loss_fake, prog_bar=False, logger=True, on_step=True)
        # self.log('texture_d_lr', lr_sch_texture_discr._last_lr[0], prog_bar=False, logger=True, on_step=True)

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        """
        Runs a single training iteration with a single batch of data.

        :param batch: batch data; combined output of dataset __getitem__ called many times (possibly output of a CollateFn)
        :param batch_idx: batch index
        :return: None
        """
        imgs_orig, masks, masked_imgs = batch  # type: Tensor, Tensor, Tensor

        # =========================================================================================
        # TRAIN GENERATOR

        self.generator.train()
        self.tile_discriminator.eval()
        self.texture_discriminator.eval()

        ideal_D_output_real, ideal_D_output_fake = self.build_tile_discriminator_ground_truths(imgs_orig, masks)
        real_texture = self.dataset.generate_texture_sample()
        fake_texture = self.generate_texture_sample()  # TODO: implement

        G_output = self.train_generator(imgs_orig=imgs_orig.detach().clone().requires_grad_(True),
                                        masked_imgs=masked_imgs,
                                        ideal_discr_output_real=ideal_D_output_real.detach().clone().requires_grad_(True),
                                        generated_texture=fake_texture.detach())
        self.generator.eval()

        # =========================================================================================
        # TRAIN TILE DISCRIMINATOR

        self.tile_discriminator.train()
        G_output = G_output.detach().clone().requires_grad_(True)
        self.train_tile_discriminator(imgs_orig=imgs_orig, ideal_tile_discr_output_real=ideal_D_output_real, ideal_tile_discr_output_fake=ideal_D_output_fake, gen_output=G_output)
        self.tile_discriminator.eval()

        # =========================================================================================
        # TRAIN TEXTURE DISCRIMINATOR

        self.texture_discriminator.train()
        self.train_texture_discriminator(fake_texture, real_texture)

        # =========================================================================================

        # log extra data a few times per epoch
        if batch_idx % (self.trainer.num_training_batches // 5) == 0:
            self.log_results_grid(batch_idx)
            self.log_texture_demo(batch_idx)

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.adam_b1, self.adam_b2))
        opt_D = torch.optim.Adam(self.tile_discriminator.parameters(), lr=self.lr, betas=(self.adam_b1, self.adam_b2))
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
            self.tile_discriminator.eval()
            sample_count = 6
            dataset = self.dataset

            # columns are original image, masked image, generator output, discriminator output
            grid = ResultsGrid(tile_rows=sample_count, tile_cols=4, tile_size=128)

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
                D_output = self.tile_discriminator(G_output)
                grid.add_tile(D_output, normalize=False)

                # metrics
                pnsr_sum += cv2.PSNR(metric_img_real, metric_img_fake)
                mean_ssim, full_ssim = structural_similarity(metric_img_real, metric_img_fake, channel_axis=2, gaussian_weights=True, use_sample_covariance=False, full=True)
                ssim_sum += mean_ssim

            grid_img = grid.get_img().copy()
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
        texture_img = builder.get_img()
        elapsed_steps = self.current_epoch * len(self.dataset) + batch_idx
        logger = self.logger  # type: TensorBoardLogger
        logger.experiment.add_image("textures", texture_img, elapsed_steps, dataformats='HWC')

        builder = TextureBuilder(self.img_size, rows, cols, self.generator, self.mask_size * 2, self.device, seed_img)
        builder.build()
        texture_img = builder.get_img()
        elapsed_steps = self.current_epoch * len(self.dataset) + batch_idx
        logger = self.logger  # type: TensorBoardLogger
        logger.experiment.add_image("textures*2", texture_img, elapsed_steps, dataformats='HWC')
