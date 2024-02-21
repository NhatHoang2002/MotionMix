# MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model

<!-- TOC -->

- [Installation](#installation)
- [Training](#prepare-environment)
- [Acknowledgement](#acknowledgement)

<!-- TOC -->

## Installation

Please refer to [install.md](install.md) for detailed installation.

## Training

Due to the requirement of a large batchsize, we highly recommend you to use DDP training.

```shell
# Baseline
python -m torch.distributed.launch --nproc_per_node=2 -m tools.train \
    --name motiondiffuse_baseline_ml3d_2gpu \
    --batch_size 128 --times 25 --num_epochs 100 --dataset_name t2m \
    --distributed

# MotionMix: This is the script we used in the paper
python -m torch.distributed.launch --nproc_per_node=2 -m tools.train \
    --name motionmix_ml3d_2gpu_lower20_upper60_ratio50_pivot60 \
    --batch_size 128 --times 25 --num_epochs 100 --dataset_name t2m \
    --t_noise_lower 20 --t_noise_upper 60 --noisy_ratio 0.5 --use_unlabeled_clean \
    --distributed
```

Besides, you can train the model on multi-GPUs with DataParallel:

```shell
# Baseline
python -m tools.train \
    --name motiondiffuse_baseline_ml3d_2gpu \
    --batch_size 128 --times 25 --num_epochs 100 --dataset_name t2m \
    --data_parallel \
    --gpu_id 0 1

# MotionMix: We never try DataParallel but we believe it should work
python -m tools.train \
    --name motionmix_ml3d_2gpu_lower20_upper60_ratio50_pivot60 \
    --batch_size 128 --times 25 --num_epochs 100 --dataset_name t2m \
    --t_noise_lower 20 --t_noise_upper 60 --noisy_ratio 0.5 --use_unlabeled_clean \
    --data_parallel \
    --gpu_id 0 1
```

Otherwise, you can run the training code on a single GPU like:

```shell
# Baseline
python -m tools.train \
    --name motiondiffuse_baseline_ml3d_1gpu \
    --batch_size 128 --times 25 --num_epochs 100 --dataset_name t2m \
    --gpu_id 0

# MotionMix
python -m tools.train \
    --name motionmix_ml3d_1gpu_lower20_upper60_ratio50_pivot60 \
    --batch_size 128 --times 25 --num_epochs 100 --dataset_name t2m \
    --t_noise_lower 20 --t_noise_upper 60 --noisy_ratio 0.5 --use_unlabeled_clean \
    --gpu_id 0
```

Here, `times` means the duplication times of the original dataset. To retain the number of iterations, you can set `times` to 25 for 1 GPU, 50 for 2 GPUs, 100 for 4 GPUs, and 200 for 8 GPUs.

Introduced arguments for MotionMix's training, including:
- `--t_noise_lower`: the lower bound $T_1$ of the noisy range $[T_1, T_2]$ in the paper
- `--t_noise_upper`: the upper bound $T_2$ of the noisy range $[T_1, T_2]$ in the paper
- `--t_noise_pivot`: the denoising pivot $T^*$ in the paper
- `--noisy_ratio`: the ratio of noisy samples in the training set, we use 50% as in the paper
- `--use_unlabeled_clean`: whether to hide the guidance annotation for the clean samples or not
 
## Evaluation

```shell
python -m tools.evaluation <model_ckpt> --eval_mode <debug|wo_mm|mm_short> --mask_at_time <time> --device_id <gpu_id>

# Baseline
python -m tools.evaluation checkpoints/t2m/motiondiffuse_baseline_ml3d_1gpu/model/latest.tar --eval_mode wo_mm --device_id 0

# MotionMix
python -m tools.evaluation checkpoints/t2m/motionmix_ml3d_2gpu_lower20_upper60_ratio50_pivot60/model/latest.tar --eval_mode wo_mm --mask_at_time 60 --device_id 0
```

Introduced argument for MotionMix's inference: `--mask_at_time` is the time step to hide the guidance condition. We use the same value with the denoising pivot `--t_noise_pivot` during training.

## Visualization

```shell
# Currently MotionDiffuse only supports visualization of models trained on the HumanML3D dataset.
# Motion length can not be larger than 196, which is the maximum length during training
# You can omit `gpu_id` to run visualization on your CPU
# Optionally, you can store the xyz coordinates of each joint to `npy_path`. The shape of motion data is (T, 22, 3), where T denotes the motion length, 22 is the number of joints.

python -m tools.visualization \
    --opt_path checkpoints/t2m/motionmix_ml3d_2gpu_lower20_upper60_ratio50_pivot60/opt.txt \
    --text "a person is jumping" \
    --motion_length 60 \
    --result_path "test_sample.gif" \
    --npy_path "test_sample.npy" \
    --mask_at_time 60 \
    --gpu_id 0
```

**Note:** You may install `matplotlib==3.3.1` to support visualization here.

## Acknowledgement

This code is developed on top of [MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model](https://github.com/mingyuan-zhang/MotionDiffuse) with the following major modifications:
- Class [`Text2MotionDataset`](datasets/dataset.py#L119):
    - Split the dataset into two subsets based on `--noisy_ratio`;
    - Approximate one subset to be noisy based on `--t_noise_lower`, `--t_noise_upper`;
    - Hide the guidance condition for the clean subset based on `--use_unlabeled_clean`.
- Method [`ScheduleSampler.sample()`](models/gaussian_diffusion.py#L59): control the sampled timesteps differently for the clean and noisy subsets, based on `--t_noise_pivot`.
- Method [`MotionTransformer.forward()`](models/transformer.py#L414): allow two-stage inference if `--mask_at_time` is set (its default value is -100).

There might be some other minor modifications in the codebase, but the above ones are the most important for MotionMix.

## Citation

If you find our work useful for your research, please consider citing our paper MotionMix and the original MotionDiffuse:
```
@misc{hoang2024motionmix,
  title={MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation}, 
  author={Nhat M. Hoang and Kehong Gong and Chuan Guo and Michael Bi Mi},
  year={2024},
  eprint={2401.11115},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{zhang2022motiondiffuse,
  title={MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model},
  author={Zhang, Mingyuan and Cai, Zhongang and Pan, Liang and Hong, Fangzhou and Guo, Xinying and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2208.15001},
  year={2022}
}