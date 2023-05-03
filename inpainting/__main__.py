"""
Entry point.
"""
import argparse
from pathlib import Path

from inpainting.params import Params
from inpainting.runner import run

arg_parser = argparse.ArgumentParser(prog='inpainting')
arg_parser.add_argument('data_dir',
                        type=Path,
                        help='Dir path to dataset location')
arg_parser.add_argument('save_dir',
                        type=Path,
                        help='Dir path for generated files')
arg_parser.add_argument('--img_size',
                        type=int,
                        required=True,
                        help='Image size, e.g. 64 for 64x64')
arg_parser.add_argument('--mask_size',
                        type=int,
                        required=True,
                        help='Size of mask border in pixels, e.g. 8')
arg_parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help='Batch size')
arg_parser.add_argument('--wandering_mask',
                        action='store_true',
                        help='Enables "wandering center patch" in masks')
arg_parser.add_argument('--exp_name',
                        type=str,
                        help='Experiment name (appears in logger, e.g. Tensorboard)')

command_group = arg_parser.add_mutually_exclusive_group()
command_group.add_argument('--dtd',
                           action='store_true',
                           help='Use DTD profile')
command_group.add_argument('--celeb',
                           action='store_true',
                           help='Use Celeb256 profile')
command_group.add_argument('--bigtex',
                           action='store_true',
                           help='Use big textures profile')
args = arg_parser.parse_args()

params = Params()
params.data_dir_path = Path(args.data_dir).resolve()
params.save_dir_path = Path(args.save_dir).resolve()

if args.dtd:
    params.dataset_name = 'dtd'
if args.celeb:
    params.dataset_name = 'celeb'
if args.bigtex:
    params.dataset_name = 'bigtex'

params.img_size = args.img_size
params.mask_size = args.mask_size
params.wandering_mask = args.wandering_mask
params.experiment_name = args.exp_name
params.batch_size = args.batch_size

print()
print(f'Dataset location:   {params.data_dir_path}')
print(f'Save data location: {params.save_dir_path}')

run(params)
