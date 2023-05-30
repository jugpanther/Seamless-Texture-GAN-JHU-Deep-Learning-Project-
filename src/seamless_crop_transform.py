"""
Transform for seamless image cropping.
"""
import numpy as np
import torch
import torchvision.transforms as T


class RandomSeamlessCrop:
    """
    Performs a random crop from a seamless image after optionally scrolling and flipping it.
    """

    def __init__(self, img_size: int, *, horiz_flip: bool, vert_flip: bool, rotate: bool):
        """
        :param img_size: size of final texture patch
        :param horiz_flip: True if horizontal flipping should be randomly applied
        :param vert_flip: True if vertical flipping should be randomly applied
        :param rotate: True if rotation should be randomly applied
        """
        self.img_size = img_size
        self.do_horiz_flip = horiz_flip
        self.do_vert_flip = vert_flip
        self.do_rotation = rotate

        self.transform = T.Compose([])
        if horiz_flip:
            self.transform.transforms.append(T.RandomHorizontalFlip())
        if vert_flip:
            self.transform.transforms.append(T.RandomVerticalFlip())
        if rotate:
            self.transform.transforms.append(T.RandomRotation((-180, 180), expand=True))
        self.transform.transforms.append(T.CenterCrop(img_size))

    @staticmethod
    def scroll_img(img: torch.Tensor, ud_scroll_pixel_count: int, lr_scroll_pixel_count: int) -> torch.Tensor:
        """
        Scrolls an image by a number of rows and columns to arrive at a new image.

        :param img: img to scroll as tensor in format ([B,]C,H,W) (the original is not modified)
        :param ud_scroll_pixel_count: number of pixels to scroll in up-down direction
        :param lr_scroll_pixel_count: number of pixels to scroll in left-right direction
        :return: new image in same format as input image
        """
        temp = torch.empty_like(img)
        ud_scroll_pixel_count %= img.shape[-2]
        lr_scroll_pixel_count %= img.shape[-1]

        if ud_scroll_pixel_count > 0:
            temp[..., :ud_scroll_pixel_count, :] = img[..., -ud_scroll_pixel_count:, :]
            temp[..., ud_scroll_pixel_count:, :] = img[..., :-ud_scroll_pixel_count, :]

        if lr_scroll_pixel_count > 0:
            temp[..., :lr_scroll_pixel_count] = img[..., -lr_scroll_pixel_count:]
            temp[..., lr_scroll_pixel_count:] = img[..., :-lr_scroll_pixel_count]

        return temp

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Performs the configured seamless crop transform.

        :param img: img to transform as tensor in format ([B,]C,H,W) (the original is not modified)
        :return: transformed image
        """
        ud_scroll_amount = np.random.randint(0, img.shape[0])
        lr_scroll_amount = np.random.randint(0, img.shape[1])
        img = self.scroll_img(img, ud_scroll_amount, lr_scroll_amount)
        img = self.transform(img)
        return img


def debug_transform():
    """
    Interactive transform debugging.
    """
    import cv2
    from src.util import tensor01_to_RGB01, download_image_as_tensor

    raw_img = download_image_as_tensor(r"https://4.bp.blogspot.com/-sMZKgoZsqkU/Us1G4Xe1r-I/AAAAAAAANLM/4f1rwRR2nZ4/s1600/Seamless_Wood_Parquet_Texture.jpg")
    assert raw_img is not None, 'Download failed'

    raw_img = torch.Tensor(raw_img).unsqueeze(0)
    raw_img = torch.permute(raw_img, (0, 3, 1, 2))
    raw_img = raw_img / 255.0
    assert raw_img.shape[0] == 1
    assert raw_img.shape[1] == 3

    cv2.imshow('Raw image', tensor01_to_RGB01(raw_img))
    cv2.waitKey(0)

    img = RandomSeamlessCrop.scroll_img(raw_img, ud_scroll_pixel_count=raw_img.shape[2] // 2, lr_scroll_pixel_count=0)
    cv2.imshow('50% up-down scroll', tensor01_to_RGB01(img))
    cv2.waitKey(0)

    img = RandomSeamlessCrop.scroll_img(raw_img, ud_scroll_pixel_count=0, lr_scroll_pixel_count=raw_img.shape[3] // 2)
    cv2.imshow('50% left-right scroll', tensor01_to_RGB01(img))
    cv2.waitKey(0)

    transform = RandomSeamlessCrop(256, horiz_flip=True, vert_flip=True, rotate=True)
    cv2.imshow('Full transform', tensor01_to_RGB01(transform(raw_img)))
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    debug_transform()
