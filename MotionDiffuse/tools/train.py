import os
import pickle
from os.path import join as pjoin

import torch
import torch.distributed as dist
import utils.paramUtil as paramUtil
from datasets import Text2MotionDataset
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from models import MotionTransformer
from options.train_options import TrainCompOptions
from trainers import DDPMTrainer
from utils.plot_script import *


def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()
    rank, world_size = get_dist_info()

    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.cached_data_path = pjoin(opt.save_root, 'cached_data.pkl')

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if opt.dataset_name == 't2m':
        opt.data_root = './data/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')

    encoder = build_models(opt, dim_pose)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    elif opt.data_parallel:
        encoder = MMDataParallel(
            encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
    else:
        encoder = encoder.cuda()

    trainer = DDPMTrainer(opt, encoder)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)

    if rank == 0 and not os.path.isfile(opt.cached_data_path):
        print(f"Data is being cached at device {rank}")
        pickle.dump(train_dataset, open(opt.cached_data_path, 'wb'))
    if world_size > 1:
        dist.barrier()
        assert os.path.isfile(opt.cached_data_path)
        print(f"Load cached data on device {rank}")
        train_dataset = pickle.load(open(opt.cached_data_path, 'rb'))

    trainer.train(train_dataset)
