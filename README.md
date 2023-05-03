# Seamless Texture GAN

by Nathan Hill & Tim Pietrzyk

---

## Requirements

📌 In addition to customary deep learning vision libraries, this code
requires [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

This code was developed and tested in Python 3.10.10 with Anaconda on Windows 10 64-bit. No other Python versions or
package managers have been tested. Refer to the `env.yaml` file for dependencies. All dependencies are "well known" and
are easily found in popular channels.

This code was developed and tested on the following PyTorch/CUDA versions:

```txt
PyTorch version:        2.0.0
Torchvision version:    0.15.0
CUDA version:           11.8
```

⚠️ **WARNING:** based on your specific hardware, drivers, and OS, it is possible (even probable) that your system may
require different versions to run this code.

---

## Setup

Create a new Anaconda environment using the supplied YAML file.

⚠️️**WARNING:** The following command may install versions of PyTorch libraries that are different from your specific
requirements due to your specific enviornment.

```commandline
conda env create -f env.yaml -p /path/to/new/env/seamlessGAN
```

A new environment named `seamlessGAN` will be created. Dependencies will be downloaded and installed automatically (an
internet connection is required). Activate the environment.

```commandline
conda activate seamlessGAN
```

## Usage

In summary, run with `?> python -m inpainting <args>`.

```commandline
usage: inpainting [-h] --img_size IMG_SIZE --mask_size MASK_SIZE --batch_size BATCH_SIZE [--wandering_mask] [--exp_name EXP_NAME] [--dtd | --celeb | --bigtex] data_dir save_dir

positional arguments:
  data_dir              Dir path to DTD dataset
  save_dir              Dir path for generated files

options:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE   Image size, e.g. 64 for 64x64
  --mask_size MASK_SIZE
                        Size of mask border in pixels, e.g. 8
  --batch_size BATCH_SIZE
                        Batch size
  --wandering_mask      Enables "wandering center patch" in masks
  --exp_name EXP_NAME   Experiment name (appears in logger, e.g. Tensorboard)
  --dtd                 Use DTD profile
  --celeb               Use Celeb256 profile
  --bigtex              Use big textures profile

```

`data_dir` is the folder that contains training images:

- For `--dtd`, provide top level DTD folder
- For `--celeb`, provide folder that contains all training images.
- For `--bigtex`, provide folder that contains all training images.

Intermediate files (like checkpoint files), Tensorboard output, and final saved model weights are saved to `save_dir`.

📌 To view live output, point Tensorboard to `/path/to/save/dir` , e.g. `tensorboard --logdir /path/to/save/dir`.

⚠️️**WARNING:** Depending on your specific dataset, you may wish to modify the augmentation definitions in `params.py`.
For example, you may want to disable rotations when learning structured textures.

### Usage example

```commandline
?> python -m inpainting --bigtex --img_size=128 --mask_size=16 --batch_size=80 --exp_name="grass_size=128_mask=16" --wandering_mask /path/to/data/dir /path/to/save/dir
```

This example runs the module in `--bigtex` mode, which loads all of the images located at `/path/to/data/dir`. All
images are 128x128 pixels, and the mask border size is 16 pixels. The batch size is 80, and the "wandering mask" is
enabled. The name of the experiment that will display in Tensorboard is `grass_size=128_mask=16`. If this experiment
already exists, a new version will be created under that experiment. All saved files, including Tensorboard data,
will be saved to `path/to/save/dir`. The relevent augmentations defined in `params.py` will be executed during training.


