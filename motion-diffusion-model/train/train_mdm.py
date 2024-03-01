# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import json
import os
import pickle

import torch
import torch.distributed as dist
from data_loaders.get_data import get_dataset, get_dataset_loader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from train.train_platforms import (  # required for the eval operation
    ClearmlPlatform, NoPlatform, TensorboardPlatform)
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args


def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite and args.resume_checkpoint == "":
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device, args.distributed)
    rank, world_size = dist_util.get_dist_info()

    print(f"creating data loader for dataset {args.dataset}...")
    dataset, collate_fn = get_dataset(name=args.dataset, num_frames=args.num_frames, args=args)
    cached_data_path = os.path.join(args.save_dir, 'cached_data.pkl')

    if dist_util.is_main_process() and not os.path.isfile(cached_data_path):
        print(f"Data is being cached at device {rank}")
        pickle.dump(dataset, open(cached_data_path, 'wb'))
    if world_size > 1:
        dist.barrier()

    assert os.path.isfile(cached_data_path)
    print(f"Loading cached data on device {rank}")
    dataset = pickle.load(open(cached_data_path, 'rb'))

    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    data = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=8, drop_last=True, collate_fn=collate_fn, sampler=sampler
    )

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data, sampler).run_loop()
    train_platform.close()

    del train_platform
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
