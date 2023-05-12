"""
Seamless texture visualization.
"""
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from src.util import channelwise_normalize_, tensor01_to_RGB01


class TextureBuilder:
    """
    Image builder class: creates an image composed of many texture tiles.
    """

    def __init__(self,
                 generator_input_size: int,
                 tile_rows: int,
                 tile_cols: int,
                 generator: nn.Module,
                 border_size: int,
                 device: any,
                 seed_img: np.ndarray):
        """
        :param generator_input_size: generator input image size
        :param tile_rows: number of rows of tiles in final image
        :param tile_cols: number of columns of tiles in final image
        :param generator: trained generator
        :param border_size: size (in pixels) of border taken from neighboring tiles
        :param device: CUDA device of generator (can be any valid device)
        :param seed_img: initial real image used to "seed" all remaining images produced by generator; must be tile-sized
        """
        if tile_rows < 2 or tile_cols < 2:
            raise ValueError('Expected the number of tile rows and cols to each be 2 or more.')

        self.generator_input_size = generator_input_size
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.generator = generator
        self.device = device
        self.border_size = border_size
        self.true_tile_size = self.generator_input_size - 2 * self.border_size
        self.img = np.random.random((self.true_tile_size * tile_rows, self.true_tile_size * tile_cols, 3))  # populate initial noise
        self.tile_order = self._randomize_tile_order()
        self._insert_seed_img(seed_img)

    def _randomize_tile_order(self) -> List[Tuple[int, int]]:
        """
        Gets a random ordering of each tile in the texture as a list of (row,col) indexes.

        :return: list of tuples in format (row,col)
        """
        tile_positions = [(row, col) for row in range(self.tile_rows) for col in range(self.tile_cols)]
        np.random.shuffle(tile_positions)
        return tile_positions

    def _insert_seed_img(self, seed_img: np.ndarray) -> None:
        """
        Inserts the seed image in a random location.

        :return: None
        """
        if seed_img.shape[0] != self.true_tile_size or seed_img.shape[1] != self.true_tile_size:
            raise ValueError(f'Seed image has shape {seed_img.shape} but expected a side length of {self.true_tile_size}')

        tile_row, tile_col = self.tile_order.pop()
        start_row = tile_row * self.true_tile_size
        end_row = start_row + self.true_tile_size
        start_col = tile_col * self.true_tile_size
        end_col = tile_col + self.true_tile_size

        self.img[start_row:end_row, start_col:end_col] = seed_img

    def get_img(self) -> np.ndarray:
        """
        Gets the created image.

        :return: image as numpy array
        """
        return self.img

    @torch.no_grad()
    def build(self) -> None:
        """
        Constructs the large texture image.

        :return: None
        """
        self.generator.eval()

        for tile_row, tile_col in self.tile_order:
            # pixels are COPIED from the large texture and PASTED into a single tile for input to the generator
            # "copy" coordinates are respective to the large texture image
            start_row_copy = max(0, tile_row * self.true_tile_size - self.border_size)
            end_row_copy = min(self.img.shape[0], (tile_row + 1) * self.true_tile_size + self.border_size)
            start_col_copy = max(0, tile_col * self.true_tile_size - self.border_size)
            end_col_copy = min(self.img.shape[1], (tile_col + 1) * self.true_tile_size + self.border_size)

            # "paste" coordinates are respective to a single (generator-input-shaped) tile
            start_row_paste = 0 if tile_row > 0 else self.border_size
            end_row_paste = self.generator_input_size if tile_row < self.tile_rows - 1 else self.generator_input_size - self.border_size
            start_col_paste = 0 if tile_col > 0 else self.border_size
            end_col_paste = self.generator_input_size if tile_col < self.tile_cols - 1 else self.generator_input_size - self.border_size

            # big texture already has default noise, but tiles around border of texture don't get all pixels copied,
            # so need to populate tile with noise first
            masked_img = np.random.random((self.generator_input_size, self.generator_input_size, 3))
            masked_img[start_row_paste:end_row_paste, start_col_paste:end_col_paste] = self.img[start_row_copy:end_row_copy, start_col_copy:end_col_copy]

            masked_img = np.transpose(masked_img, (2, 0, 1))
            channelwise_normalize_(masked_img, 0, 0.5)
            masked_img = torch.tensor(masked_img, dtype=torch.float32).unsqueeze(0).to(self.device)
            G_output = self.generator(masked_img)
            G_output = tensor01_to_RGB01(G_output).squeeze(0)

            self.img[start_row_copy:end_row_copy, start_col_copy:end_col_copy] = G_output[start_row_paste:end_row_paste, start_col_paste:end_col_paste]
