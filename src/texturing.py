"""
Seamless texture visualization.
"""

import numpy as np
import torch
from torch import nn

from src.util import channelwise_normalize_, tensor01_to_RGB01


class TextureBuilder:
    """
    Image builder class: creates an image composed of many texture tiles.
    """

    def __init__(self, generator_size: int, tile_rows: int, tile_cols: int, generator: nn.Module, border_size: int, device: any, seed_img: np.ndarray):
        """
        :param generator_size: generator input image size
        :param tile_rows: number of rows in final image
        :param tile_cols: number of columns in final image
        :param generator: trained generator
        :param border_size: size (in pixels) of boarder taken from neighboring tiles
        :param device: CUDA device of generator (can be any valid device)
        :param seed_img: initial real image used to "seed" all remaining images produced by generator
        """
        if tile_rows < 2 or tile_cols < 2:
            raise ValueError('Expected the number of tile rows and cols to each be 2 or more.')

        self.generator_size = generator_size
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.generator = generator
        self.device = device
        self.border_size = border_size
        self.output_crop_size = self.generator_size - 2 * self.border_size
        self.img = np.random.random((self.output_crop_size * tile_rows, self.output_crop_size * tile_cols, 3))
        self.img[:seed_img.shape[0], :seed_img.shape[1]] = seed_img

    def get(self) -> np.ndarray:
        """
        Gets the created image.

        :return: image as numpy array
        """
        return self.img

    def build(self) -> None:
        """
        Constructs the large texture image.

        :return: None
        """
        with torch.no_grad():
            self.generator.eval()

            for tile_row in range(self.tile_rows):
                for tile_col in range(1 if tile_row == 0 else 0, self.tile_cols):
                    # "copy" coordinates are respective to the large grid image
                    start_row_copy = max(0, tile_row * self.output_crop_size - self.border_size)
                    end_row_copy = min(self.img.shape[0], (tile_row + 1) * self.output_crop_size + self.border_size)
                    start_col_copy = max(0, tile_col * self.output_crop_size - self.border_size)
                    end_col_copy = min(self.img.shape[1], (tile_col + 1) * self.output_crop_size + self.border_size)

                    # "paste" coordinates are respective to a single (generator-input-shaped) tile
                    start_row_paste = 0 if tile_row > 0 else self.border_size
                    end_row_paste = self.generator_size if tile_row < self.tile_rows - 1 else self.generator_size - self.border_size
                    start_col_paste = 0 if tile_col > 0 else self.border_size
                    end_col_paste = self.generator_size if tile_col < self.tile_cols - 1 else self.generator_size - self.border_size

                    masked_img = np.random.random((self.generator_size, self.generator_size, 3))  # because we can't copy all img pixels around edge of large image
                    masked_img[start_row_paste:end_row_paste, start_col_paste:end_col_paste] = self.img[start_row_copy:end_row_copy, start_col_copy:end_col_copy]

                    masked_img = np.transpose(masked_img, (2, 0, 1))
                    channelwise_normalize_(masked_img, 0, 0.5)
                    masked_img = torch.tensor(masked_img, dtype=torch.float32).unsqueeze(0).to(self.device)
                    G_output = self.generator(masked_img)
                    G_output = tensor01_to_RGB01(G_output).squeeze(0)

                    self.img[start_row_copy:end_row_copy, start_col_copy:end_col_copy] = G_output[start_row_paste:end_row_paste, start_col_paste:end_col_paste]
