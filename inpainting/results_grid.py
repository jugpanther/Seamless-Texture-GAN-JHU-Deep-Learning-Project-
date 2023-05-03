"""
Grid image generation.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from inpainting.util import channelwise_normalize_, tensor01_to_RGB01


class ResultsGrid:
    """
    Represents a grid of result imagery.
    Col 1: original input image after transformations
    Col 2: image mask
    Col 3: generator output (painted image)
    Col 4: discriminator output (pixel-wise probabilities)
    """

    def __init__(self, *,
                 rows: int,
                 cols: int,
                 tile_size: int,
                 padding: int = 4,
                 bg_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize an empty results grid object.
        :param rows: number of tile rows
        :param cols: number of tile columns
        :param tile_size: edge length in pixels of a single tile (e.g. 128)
        :param padding: padding separation
        :param bg_color: background color
        """
        self.rows = rows
        self.cols = cols
        self.padding = padding
        self.img_size = tile_size
        self.grid = np.ones((self.rows * (self.img_size + padding) + padding, self.cols * (self.img_size + padding) + padding, 3))
        if max(bg_color) > 1.0:
            raise ValueError('All background color values must be in range [0,1].')
        for i in range(3):
            self.grid[:, :, i] *= bg_color[i]
        self.current_tile_row_index = 0
        self.current_tile_col_index = 0
        self.total_img_count = self.rows * self.cols
        self.required_tile_shape = (self.img_size, self.img_size, 3)

    def add_tile(self, tile_tensor: Tensor, normalize: bool = False) -> None:
        """
        Adds the next tile to the grid. Caller is responsible for tracking the current row and col this tile will occupy.
        :param tile_tensor: 3-channel image tensor; can contain batch dimension; general expected format is ([B,] 3, H, W)
        :param normalize: if True, performs channel-wise normalization on the image
        """
        if self.current_tile_row_index > self.rows:
            raise ValueError('Grid is already full')

        if len(tile_tensor.shape) == 4:
            tile_tensor = tile_tensor[0]

        if tile_tensor.shape[0] == 1:
            tile_tensor = torch.vstack((tile_tensor, tile_tensor, tile_tensor))

        if normalize:
            channelwise_normalize_(tile_tensor, 255 // 2, 30)

        self._add_tile(self._resize_rgb_array(tensor01_to_RGB01(tile_tensor)))

    def add_blank_tile(self) -> None:
        """
        Adds a blank tile to the grid. Drawn as a black square with white 'X'. Represents missing or invalid data.
        Caller is responsible for tracking the current row and col this tile will occupy.

        :return: None
        """
        x = np.eye(self.img_size, self.img_size)
        x = x + np.rot90(x)
        x = np.dstack((x, x, x))
        x = np.transpose(x, (2, 1, 0))
        x = torch.Tensor(x)
        self.add_tile(x)

    def _add_tile(self, tile_img: np.ndarray) -> None:
        """
        Add a single preprocessed tile sub-image to the grid.
        :param tile_img: image
        :return: None
        """
        assert tile_img.shape == self.required_tile_shape, f'Tile image must have shape {self.required_tile_shape}; found shape {tile_img.shape}'

        start_row = self.current_tile_row_index * (self.img_size + self.padding) + self.padding
        end_row = start_row + self.img_size
        start_col = self.current_tile_col_index * (self.img_size + self.padding) + self.padding
        end_col = start_col + self.img_size

        self.grid[start_row: end_row, start_col: end_col, :] = tile_img

        self.current_tile_col_index += 1
        if self.current_tile_col_index >= self.cols:
            self.current_tile_col_index = 0
            self.current_tile_row_index += 1

        self.total_img_count += 1

    def _resize_rgb_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Resizes an 3-channel array representing an image. The original data is not modified.
        :param arr: array to convert
        :return: array
        """
        return cv2.resize(arr, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

    def get(self) -> np.ndarray:
        """
        Returns the complete grid image.
        :return: image
        """
        return self.grid
