from typing import Tuple

import numpy as np
import torchvision.transforms as T

from src.runner import ImageSamplingProfiles
from src.seamless_crop_transform import RandomSeamlessCrop


class AugTypes:
    FULL = 0
    NO_ROT = 1
    VERT_FLIP_ONLY = 2
    HORIZ_FLIP_ONLY = 3
    NONE = 4

    @staticmethod
    def as_str(aug_type: int) -> str:
        """
        Stringifies a given augmentation type.

        :param aug_type: type to stringify
        :return: str
        """
        if aug_type == AugTypes.FULL:
            return 'FULL'

        elif aug_type == AugTypes.NO_ROT:
            return 'NO_ROT'

        elif aug_type == AugTypes.VERT_FLIP_ONLY:
            return 'VERT_FLIP_ONLY'

        elif aug_type == AugTypes.HORIZ_FLIP_ONLY:
            return 'HORIZ_FLIP_ONLY'

        elif aug_type == AugTypes.NONE:
            return 'NONE'

        else:
            raise ValueError(f'Unknown augmentation type: {aug_type}')


def get_transform(profile: str, aug_type: int, final_img_size: int) -> Tuple[T.Compose, T.Compose]:
    """
    Builds transforms for augmentation of the desired type to produce the correct output image size.
    The "base transform" is applied to each image upon reading from disk.
    The "aug transform" is applied to modify the image or extract some sub-image to produce an augmentation.
    The "texture transform" is used to extract a large texture patch from a source image.

    :param profile: current run profile
    :param aug_type: aug type
    :param final_img_size: final image size, e.g. 64 (assumes square images)
    :return: transform.Compose objects as tuple: (base transform, aug transform, texture transform)
    """
    base_transform = _get_base_transform(profile, final_img_size)

    if aug_type == AugTypes.FULL:
        transforms = _get_full_transform(final_img_size), RandomSeamlessCrop(final_img_size, horiz_flip=True, vert_flip=True, rotate=True)

    elif aug_type == AugTypes.NO_ROT:
        transforms = _get_no_rot_transform(base_transform), RandomSeamlessCrop(final_img_size, horiz_flip=True, vert_flip=True, rotate=False)

    elif aug_type == AugTypes.VERT_FLIP_ONLY:
        transforms = _get_vert_flip_transform(base_transform), RandomSeamlessCrop(final_img_size, horiz_flip=False, vert_flip=True, rotate=False)

    elif aug_type == AugTypes.HORIZ_FLIP_ONLY:
        transforms = _get_horiz_flip_transform(base_transform), RandomSeamlessCrop(final_img_size, horiz_flip=True, vert_flip=False, rotate=False)

    elif aug_type == AugTypes.NONE:
        transforms = _get_empty_transform(), _get_empty_transform()

    else:
        raise ValueError(f'Unknown augmentation type: {aug_type}')

    return base_transform, *transforms


def _get_base_transform(profile: str, final_img_size: int) -> T.Compose:
    if profile == ImageSamplingProfiles.SUBSAMPLING:
        return T.Compose([T.RandomCrop(final_img_size)])
    elif profile == ImageSamplingProfiles.SINGLE_SAMPLE:
        return T.Compose([T.Resize(final_img_size, T.InterpolationMode.BILINEAR)])


def _get_full_transform(final_img_size: int) -> T.Compose:
    initial_img_size = final_img_size * np.sin(np.pi / 4) * 2
    return T.Compose([
        T.RandomCrop(initial_img_size),
        T.RandomRotation((-180, 180), expand=True),
        T.CenterCrop((final_img_size, final_img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])


def _get_no_rot_transform(base_transform) -> T.Compose:
    return T.Compose([
        base_transform,
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])


def _get_vert_flip_transform(base_transform) -> T.Compose:
    return T.Compose([
        base_transform,
        T.RandomVerticalFlip()
    ])


def _get_horiz_flip_transform(base_transform) -> T.Compose:
    return T.Compose([
        base_transform,
        T.RandomHorizontalFlip()
    ])


def _get_empty_transform() -> T.Compose:
    return T.Compose([])
