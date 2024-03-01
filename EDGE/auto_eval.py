import json
import os
import pickle
import time
from pathlib import Path
from test import test

import numpy as np
import torch
from args import parse_test_opt
# from eval.beat_align_score import calc_ba_score
from eval.eval_pfc import calc_physical_score
# from eval.metrics_new_axm import calc_and_save_metrics, quantized_metrics
from vis import SMPLSkeleton


def get_metric_statistics(values):
    metrics = np.array(values)
    mean = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    conf_interval = 1.96 * std / np.sqrt(len(metrics))
    return mean, conf_interval


def evaluation(opt):
    final_metrics = {}
    all_metrics = {
        "pfc": [],
        # "beat_alignment": [],
        # "fid_k": [],
        # "fid_m": [],
        # "div_k": [],
        # "div_m": [],
        # "pfc_gt": [],
        # "beat_alignment_gt": [],
        # "div_k_gt": [],
        # "div_m_gt": [],
    }

    for i in range(opt.replication_times):
        print("[ Start replication {} ]".format(i+1))
        print("[ Generating motions ]")
        test(opt)

        print("[ Calculating PFC ]")
        pfc = calc_physical_score(opt.motion_save_dir)
        # gt_pfc = calc_physical_score(opt.gt_motion)

        # print("[ Calculating beat alignment ]")
        # ba = calc_ba_score(opt, opt.motion_save_dir)
        # gt_beat_score = calc_ba_score(opt, opt.gt_motion)

        # print("[ Calculating FID, Div ]")
        # calc_and_save_metrics(opt.gt_motion)
        # calc_and_save_metrics(opt.motion_save_dir)
        # fid_div_rlt = quantized_metrics(opt.motion_save_dir, opt.gt_motion)

        all_metrics["pfc"].append(pfc)
        # all_metrics["beat_alignment"].append(ba)
        # all_metrics["fid_k"].append(fid_div_rlt["fid_k"])
        # all_metrics["fid_m"].append(fid_div_rlt["fid_m"])
        # all_metrics["div_k"].append(fid_div_rlt["div_k"])
        # all_metrics["div_m"].append(fid_div_rlt["div_m"])
        # all_metrics["pfc_gt"].append(gt_pfc)
        # all_metrics["beat_alignment_gt"].append(gt_beat_score)
        # all_metrics["div_k_gt"].append(fid_div_rlt["div_k_gt"])
        # all_metrics["div_m_gt"].append(fid_div_rlt["div_m_gt"])

        for metric, scores in all_metrics.items():
            mean, conf_interval = get_metric_statistics(scores)
            print(f'---> {metric}: {mean:.3f} CInterval: {conf_interval:.3f}')

            final_metrics[metric] = f'{mean:.3f} +- {conf_interval:.3f}'

        final_metrics["replications"] = i+1
        with open(opt.out_dir, "w") as fp:
            json.dump(final_metrics, fp, indent=4)


def calculate_gt_feats(root='./data/test/motions', save_dir=''):
    smpl = SMPLSkeleton()
    for pkl in os.listdir(root):
        outname = f"test_{pkl.split('.')[0]}.pkl"
        if os.path.isdir(os.path.join(root, pkl)):
            continue

        data = np.load(os.path.join(root, pkl), allow_pickle=True)
        q = torch.from_numpy(data['q'].reshape(-1, 24, 3)).unsqueeze(0)
        pos = torch.from_numpy(data['pos']).unsqueeze(0)
        full_pose = smpl.forward(q, pos).detach().cpu().numpy()

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump(
            {
                "smpl_poses": q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                "smpl_trans": pos.squeeze(0).cpu().numpy(),
                "full_pose": full_pose[0],
            },
            open(os.path.join(save_dir, outname), "wb"),
        )

if __name__ == "__main__":
    start_time = time.time()

    opt = parse_test_opt()
    opt.replication_times = 20
    opt.gt_motion = './eval/data4eval/aist_gt_motion'

    exp_name = opt.checkpoint.split('/')[-3]
    epoch = opt.checkpoint.split('/')[-1].split('-')[1].split('.')[0]

    os.makedirs(f"./runs/test/{exp_name}", exist_ok=True)
    opt.out_dir = f"./runs/test/{exp_name}/eval_e{epoch}_result.json"

    evaluation(opt)
