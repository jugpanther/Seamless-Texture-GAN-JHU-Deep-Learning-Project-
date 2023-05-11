"""
Utility functions.
"""

import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision


def is_power_of_2(value: int) -> bool:
    """
    Determines if a value is a positive power of 2.
    :param value: positive integer
    :return: True if power of 2, else False
    """
    return value > 0 and (value & (value - 1) == 0)


def print_layer_sizes(model: nn.Module, input_size: torch.Size | Tuple[int, ...]) -> None:
    """
    Given an input tensor size, prints the output shape for every convolutional layer in the model
    (Conv2d or ConvTranspose2d).
    :param model: model to inspect
    :param input_size: tensor size; do not include batch dimension, e.g. (3, 64, 64)
    :return: None; prints output
    """
    print('Printing output sizes for all convolutional layers...')
    conv_counter = 1
    x = torch.rand(1, *input_size)
    for i, layer in enumerate(model.children()):
        out = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            name = ' Conv' if isinstance(layer, nn.Conv2d) else 'TConv'
            print(f'{name} #{conv_counter} (layer {i}): {x.shape} -> {out.shape}')
            conv_counter += 1
        x = out
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {param_count} ({param_count / 1000000:.3f} million)')


def model_sanity_check(model: nn.Module, input_size: torch.Size | Tuple[int, ...], output_size: torch.Size | Tuple[int, ...], model_name: str = 'model') -> None:
    """
    Performs a sanity check on a model to verify the desired output size is produced for a given input size.
    Raises an exception if the desired output size does not match the observed output size.
    :param model: model to check
    :param input_size: size of input tensor (single instance), e.g. torch.Size(3,256,256). Do not include batch dimension.
    :param output_size: desired size of output tensor (single instance), e.g. torch.Size(3,256,256) or torch.Size(1000). Do not include batch size.
    :param model_name: optional; name of model
    :return: None
    """
    if not isinstance(input_size, (torch.Size, Tuple)) or not isinstance(output_size, (torch.Size, Tuple)):
        raise ValueError(f'input and output sizes must each be type torch.Size or tuple(int,...), not {type(input_size)} and {type(output_size)}')

    input_tensor = torch.rand(1, *input_size)
    output_tensor = model.forward(input_tensor)
    desired_output_shape = torch.Size((1, *output_size))

    if output_tensor.shape != desired_output_shape:
        raise ValueError(f'{model_name} output size mismatch: given input size of {input_tensor.shape}, output size is {output_tensor.shape} but should be {output_size}')


def channelwise_normalize_(img: torch.Tensor | np.ndarray, new_mean: float, new_std: float) -> None:
    """
    Normalizes each channel of an image IN PLACE to have a given mean and std.
    :param img: img array in format (C,H,W) or (H,W,C)
    :param new_std: desired mean
    :param new_mean: desired std
    :return: None, tensor is modified in place
    """
    # https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
    assert new_std > 0, f'Standard deviation must be greater than zero; found {new_std}'
    if img.shape[0] in [1, 3]:
        for c in range(img.shape[0]):
            img[c] = new_mean + (img[c] - img[c].mean()) * new_std / img[c].std()
    elif img.shape[2] in [1, 3]:
        for c in range(img.shape[2]):
            img[..., c] = new_mean + (img[..., c] - img[..., c].mean()) * new_std / img[..., c].std()


def tensor01_to_RGB01(t: torch.Tensor) -> np.ndarray:
    """
    Converts a tensor in format [C,H,W] or [B,C,H,W] in range [0,1]
    to an RGB array in format [H,W,C] in range [0,1].
    :param t: tensor to convert (original is not modified)
    :return: RGB array
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError(f'Expected a tensor but found {type(t)}.')

    if torch.min(t) < 0 or torch.max(t) > 1:
        torch.clip_(t, 0, 1)

    t = t.clone().detach().cpu().numpy()
    transpose_shape = (0, 2, 3, 1) if t.ndim == 4 else (1, 2, 0)
    t = np.transpose(t, transpose_shape)
    t = t.astype(np.float32)
    return t


def get_cuda_device() -> any:
    """
    Gets the current CUDA device.
    :return: any
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 'cpu'


def print_torch_version_info() -> None:
    """
    Prints version data and other technical info for PyTorch and CUDA.
    :return: None
    """
    print(f'Python version: {sys.version}')
    print(f'PyTorch Version: {torch.__version__}')
    print(f'Torchvision Version: {torchvision.__version__}')
    print(f'CUDA version: {torch.version.cuda}')

    device = get_cuda_device()

    if torch.cuda.is_available():
        print(f'{torch.cuda.device_count()} CUDA device(s) available')
        print(f'CUDA device is {device}, which refers to "{torch.cuda.get_device_name()}"')
    else:
        print(f'CUDA is unavailable. Device is "{device}".')
