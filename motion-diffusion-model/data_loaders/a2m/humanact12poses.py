import pickle as pkl
import random
import numpy as np
import os
import torch
from .dataset import Dataset
from utils.model_util import create_gaussian_diffusion


class HumanAct12Poses(Dataset):
    dataname = "humanact12"

    def __init__(self, args, datapath="dataset/HumanAct12Poses", split="train", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "humanact12poses.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]

        self._actions = [x for x in data["y"]]

        total_num_actions = len(humanact12_coarse_action_enumerator)
        self._clean_masks = [True for _ in self._pose]
        if split == "train" and args.noisy_ratio > 0:
            assert len(self._pose) == len(self._joints)

            t_noise_lower = args.t_noise_lower
            t_noise_upper = args.t_noise_upper

            print(f"Adding noise of range [{t_noise_lower}, {t_noise_upper}] to {args.noisy_ratio * 100}% of the data")
            print(f"Hiding label for the clean data: {args.use_unlabeled_clean}")

            noisy_dict = {action: [] for action in range(total_num_actions)}
            for i, action in enumerate(self._actions):
                noisy_dict[action].append(i)
            noisy_dict = {
                action: random.sample(indices, int(len(indices) * args.noisy_ratio))
                for action, indices in noisy_dict.items()
            }
            aug_indices = [index for action, indices in noisy_dict.items() for index in indices]

            diffusion = create_gaussian_diffusion(args)

            for i in aug_indices:
                self._clean_masks[i] = False

                _t_noise = torch.randint(t_noise_lower, t_noise_upper, (1,))

                self._pose[i] = torch.from_numpy(self._pose[i]).float()
                self._pose[i] = diffusion.q_sample(self._pose[i].unsqueeze(0), _t_noise)
                self._pose[i] = self._pose[i].squeeze().detach().numpy()

                self._joints[i] = torch.from_numpy(self._joints[i]).float()
                self._joints[i] = diffusion.q_sample(self._joints[i].unsqueeze(0), _t_noise)
                self._joints[i] = self._joints[i].squeeze().detach().numpy()
            
            if args.use_unlabeled_clean:
                self._actions = [12 if self._clean_masks[i] else self._actions[i] for i in range(len(self._actions))]
            
            del diffusion
        else:
            print("Use all annotated clean data for training")

        self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = humanact12_coarse_action_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose

    def __getitem__(self, index):
        if self.split == "train":
            data_index = self._train[index]
        else:
            data_index = self._test[index]
        output = super()._get_item_data_index(data_index)
        output["clean_mask"] = self._clean_masks[data_index]
        return output

humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
    12: "<unk>"
}
