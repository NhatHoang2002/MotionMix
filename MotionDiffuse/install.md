# Installation

This will describe the environment setup and data preparation for the MotionMix project, visit here if you need for for [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse/blob/main/text2motion/install.md).

- [Requirements](#requirements)
- [Prepare environment](#prepare-environment)
- [Data Preparation](#data-preparation)

<!-- TOC -->

## Requirements

MotionMix was trained and tested on the following:
- Linux
- Python 3.7+
- PyTorch 1.10.1.
- CUDA 12.2
- [MMCV](https://github.com/open-mmlab/mmcv) 1.5.3 (Please install mmcv-full>=1.3.17,<1.6.0 for GPU as mentioned in MotionDiffuse)

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n motiondiffuse python=3.7 -y
conda activate motiondiffuse
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
```shell
conda install pytorch={torch_version} torchvision cudatoolkit={cu_version} -c pytorch
```

E.g., install PyTorch 1.7.1 & CUDA 10.1.
```shell
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch
```

**Important:** Make sure that your compilation CUDA version and runtime CUDA version match.

c. Build mmcv-full

- mmcv-full

We recommend you to install the pre-build package as below.

For CPU:
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/{torch_version}/index.html
```
Please replace `{torch_version}` in the url to your desired one.

For GPU:
```shell
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
Please replace `{cu_version}` and `{torch_version}` in the url to your desired one.

For example, to install mmcv-full with CUDA 10.1 and PyTorch 1.7.1, use the following command:
```shell
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.1/index.html
```

See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
For more version download link, refer to [openmmlab-download](https://download.openmmlab.com/mmcv/dist/index.html).


d. Install other requirements

```shell
pip install -r requirements.txt
```

## Data Preparation

a. Download datasets

For both the HumanML3D dataset and the KIT-ML dataset, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).

b. Download pretrained weights for evaluation

We use the same evaluation protocol as [this repo](https://github.com/EricGuo5513/text-to-motion). You should download pretrained weights of the contrastive models in [t2m](https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view) and [kit](https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view) for calculating FID and precisions. To dynamically estimate the length of the target motion, `length_est_bigru` and [Glove data](https://drive.google.com/drive/folders/1qxHtwffhfI4qMwptNW6KJEDuT6bduqO7?usp=sharing) are required.

## Download Checkpoints

a. Download pretrained weights for **MotionDiffuse**

The pretrained weights for the baseline MotionDiffuse can be downloaded from [here](https://drive.google.com/drive/folders/1qxHtwffhfI4qMwptNW6KJEDuT6bduqO7?usp=sharing)

b. Download pretrained weights for **MotionMix**

*We will soon release the pretrained weights for MotionMix...*

---

Download the above resources and arrange them in the following file structure:
```text
MotionDiffuse
├── checkpoints
│   ├── kit
│   │   └── kit_motiondiffuse
│   │       ├── meta
│   │       │   ├── mean.npy
│   │       │   └── std.npy
│   │       ├── model
│   │       │   └── latest.tar
│   │       └── opt.txt
│   └── t2m
│       └── t2m_motiondiffuse
│           ├── meta
│           │   ├── mean.npy
│           │   └── std.npy
│           ├── model
│           │   └── latest.tar
│           └── opt.txt
└── data
    ├── glove
    │   ├── our_vab_data.npy
    │   ├── our_vab_idx.pkl
    │   └── out_vab_words.pkl
    ├── pretrained_models
    │   ├── kit
    │   │   └── text_mot_match
    │   │       └── model
    │   │           └── finest.tar
    │   └── t2m
    │   │   ├── text_mot_match
    │   │   │   └── model
    │   │   │       └── finest.tar
    │   │   └── length_est_bigru
    │   │       └── model
    │   │           └── finest.tar
    ├── HumanML3D
    │   ├── new_joint_vecs
    │   │   └── ...
    │   ├── new_joints
    │   │   └── ...
    │   ├── texts
    │   │   └── ...
    │   ├── Mean.npy
    │   ├── Std.npy
    │   ├── test.txt
    │   ├── train_val.txt
    │   ├── train.txt
    │   └── val.txt
    └── KIT-ML
        ├── new_joint_vecs
        │   └── ...
        ├── new_joints
        │   └── ...
        ├── texts
        │   └── ...
        ├── Mean.npy
        ├── Std.npy
        ├── test.txt
        ├── train_val.txt
        ├── train.txt
        └── val.txt
```