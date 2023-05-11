"""
Entry point.
"""
import argparse
from pathlib import Path

from src.params import Params
from src.runner import run, SupportedProfiles
from src.augmentation import AugTypes, get_transform

arg_parser = argparse.ArgumentParser(prog='src')

arg_parser.add_argument('data_dir',
                        type=Path,
                        help='Dir path to dataset location')

arg_parser.add_argument('save_dir',
                        type=Path,
                        help='Dir path for generated files')

arg_parser.add_argument('--img_size',
                        type=int,
                        required=True,
                        help='Image size (always square), e.g. 64 for 64x64')

arg_parser.add_argument('--mask_size',
                        type=int,
                        required=True,
                        help='Size of mask border in pixels, e.g. 8')

arg_parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help='Batch size')

arg_parser.add_argument('--lr',
                        type=float,
                        required=True,
                        help='Learning rate')

arg_parser.add_argument('--wandering_mask_prob',
                        type=float,
                        help='Probability [0-1] of "wandering center patch" appearing per mask')

arg_parser.add_argument('--exp_name',
                        type=str,
                        help='Experiment name (appears in logger, e.g. Tensorboard)')

arg_parser.add_argument('--profile',
                        type=str,
                        choices=SupportedProfiles.values,  # noqa
                        help='Run profile')

arg_parser.add_argument('--samples_per_img',
                        type=int,
                        help='Number of samples taken per image')

arg_parser.add_argument('--augs_per_sample',
                        type=int,
                        help='Number of augmentations taken per sample')

aug_group = arg_parser.add_mutually_exclusive_group(required=True)

aug_group.add_argument('--full_aug',
                       action='store_true',
                       help='Full augmentation; flips and rotations')

aug_group.add_argument('--no_rot',
                       action='store_true',
                       help='No rotations; flips only')

aug_group.add_argument('--vert_flip',
                       action='store_true',
                       help='Vertical flips only')

aug_group.add_argument('--horiz_flip',
                       action='store_true',
                       help='Horizontal flips only')

aug_group.add_argument('--no_aug',
                       action='store_true',
                       help='No augmentations')

arg_parser.add_argument('--workers',
                        type=int,
                        default=1,
                        required=False,
                        help='Number of training works (<= num CPUs)')

args = arg_parser.parse_args()

params = Params()
params.data_dir_path = Path(args.data_dir).resolve()
params.save_dir_path = Path(args.save_dir).resolve()

params.img_size = args.img_size
params.mask_size = args.mask_size
params.batch_size = args.batch_size
params.wandering_mask_prob = args.wandering_mask_prob

params.experiment_name = args.exp_name
params.run_profile = args.profile
params.samples_per_img = args.samples_per_img
params.augs_per_sample = args.augs_per_sample

if args.full_aug:
    params.aug_type = AugTypes.FULL
elif args.no_rot:
    params.aug_type = AugTypes.NO_ROT
elif args.vert_flip:
    params.aug_type = AugTypes.VERT_FLIP_ONLY
elif args.horiz_flip:
    params.aug_type = AugTypes.HORIZ_FLIP_ONLY
elif args.no_aug:
    params.aug_type = AugTypes.NONE

params.base_transform, params.aug_transform = get_transform(params.run_profile, params.aug_type, params.img_size)
params.train_worker_count = args.workers

run(params)
