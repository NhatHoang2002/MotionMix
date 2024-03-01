# EDGE: Editable Dance Generation From Music

## Requirements
* 64-bit Python 3.7+
* PyTorch 1.12.1
* At least 16 GB RAM per GPU
* 1&ndash;8 high-end NVIDIA GPUs with at least 16 GB of GPU memory, NVIDIA drivers, CUDA 11.6 toolkit.
* **MotionMix** also worked with PyTorch 2.1.2 and CUDA 12.2 toolkit.

This repository additionally depends on the following libraries, which may require special installation procedures:
* [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)
* [pytorch3d](https://github.com/facebookresearch/pytorch3d)
* [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index)
	* Note: after installation, don't forget to run `accelerate config` . We use fp16.
* [wine](https://www.winehq.org) (Optional, for import to Blender only)

### Dataset Download
Download and process the AIST++ dataset (wavs and motion only) using:
```bash
cd data
bash download_dataset.sh
python create_dataset.py --extract-baseline --extract-jukebox
```
This will process the dataset to match the settings used in the paper. The data processing will take ~24 hrs and ~60 GB to precompute all the Jukebox features for the dataset.

### Train Model
Once the AIST++ dataset is downloaded and processed, run the training script, e.g.
```bash
# Baseline
accelerate launch train.py --exp_name edge_baseline --batch_size 128  --epochs 2000 --feature_type jukebox --processed_data_dir data/dataset_backups_baseline

# MotionMix (we trained with 2 GPUs in the paper)
accelerate launch train.py --exp_name motionmix_lower20_upper80_ratio50_pivot80 --batch_size 128  --epochs 2000 --feature_type jukebox --t_noise_lower 20 --t_noise_upper 80 --t_noise_pivot 80 --noisy_ratio 0.5 --use_unlabeled_clean --processed_data_dir data/dataset_backups_motionmix
```

Introduced arguments for MotionMix's training, including:
- `--t_noise_lower`: the lower bound $T_1$ of the noisy range $[T_1, T_2]$ in the paper
- `--t_noise_upper`: the upper bound $T_2$ of the noisy range $[T_1, T_2]$ in the paper
- `--t_noise_pivot`: the denoising pivot $T^*$ in the paper
- `--noisy_ratio`: the ratio of noisy samples in the training set, we use 50% as in the paper
- `--use_unlabeled_clean`: whether to hide the guidance annotation for the clean samples or not

The training will log progress to `wandb` by default and intermittently produce sample outputs to visualize learning. Please modify the code if you want to disable wandb.

### Evaluate Model
Unlike previous studies that usually presented only one evaluation result, we found metrics to be inconsistent. Therefore, we provide a more comprehensive evaluation by presenting the average and 95% confidence interval from 20 evaluation runs by providing a new file `auto_eval.py`.

```bash
# Baseline
python auto_eval.py --checkpoint ./runs/train/edge_baseline/weights/train-2000.pt --save_motions

# MotionMix
python auto_eval.py --checkpoint ./runs/train/motionmix_edge_2gpu_lower20_upper80_ratio50_pivot80/weights/train-2000.pt --save_motions --mask_at_time 80
```

Introduced argument for MotionMix's inference: `--mask_at_time` is the time step to hide the guidance condition. We use the same value with the denoising pivot `--t_noise_pivot` during training.

**Note**: the script above can only calculate PFC metric like in [EDGE]((https://github.com/Stanford-TML/EDGE)). If you want to calculate other metrics, please refer to [Bailando](https://github.com/lisiyao21/Bailando).

## Blender 3D rendering
In order to render generated dances in 3D, we convert them into FBX files to be used in Blender. We provide a sample rig, `SMPL-to-FBX/ybot.fbx`.
After generating dances with the `--save-motions` flag enabled, move the relevant saved `.pkl` files to a folder, e.g. `smpl_samples`
Run
```.bash
python SMPL-to-FBX/Convert.py --input_dir SMPL-to-FBX/smpl_samples/ --output_dir SMPL-to-FBX/fbx_out
```
to convert motions into FBX files, which can be imported into Blender and retargeted onto different rigs, i.e. from [Mixamo](https://www.mixamo.com). A variety of retargeting tools are available, such as the [Rokoko plugin for Blender](https://www.rokoko.com/integrations/blender).

## Acknowledgement

This code is developed on top of [EDGE: Editable Dance Generation From Music](https://github.com/Stanford-TML/EDGE) with the following major modifications:
- Class [`AISTPPDataset`](dataset/dance_dataset.py#L73):
    - Split the dataset into two subsets based on `--noisy_ratio`;
    - Approximate one subset to be noisy based on `--t_noise_lower`, `--t_noise_upper`;
    - Hide the guidance condition for the clean subset based on `--use_unlabeled_clean`.
- Method [`GaussianDiffusion.loss()`](model/diffusion.py#L535): control the sampled timesteps differently for the clean and noisy subsets, based on `--t_noise_pivot`.
- Method [`GaussianDiffusion.long_ddim_sample()`](model/diffusion.py#L310): allow two-stage inference if `--mask_at_time` is set (its default value is -100).

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

@article{tseng2022edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C Karen},
  journal={arXiv preprint arXiv:2211.10658},
  year={2022}
}
```
