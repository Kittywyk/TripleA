# TripleA  
This is the official PyTorch implementation for the paper "TripleA: An Unsupervised Domain Adaptation Framework for Nighttime VRU Detection".

## 1. Description
TripleA, an unsupervised domain adaptation framework is introduced to achieve nighttime VRU detection. Realized through a crucial triple alignment, TripleA first aligns the distributions of the labeled daytime domain with the unlabeled nighttime domain. Then, the degraded image is enhanced in terms of illumination
and noise. We present an illumination difference-aware denoising network to address the intractable noise and enable selfsupervised learning through a meticulously designed exchange-recombination strategy, which is integrated into a novel pseudosupervised attention to achieve noise distribution alignment.
To further enhance the capabilities of the denoising network under real-world scenarios, we introduce degradation alignment to enforce domain-invariant degradation encoding.

## 2. Create Environment
We recommend that you create two environments to reproduce the distribution alignment and image enhancement components of the framework, respectively.
### 2.1 Requirements
- Python3
- PyTorch
- opencv
- numpy

### 2.2 Set up the environment 
For distribution alignment re-implementation, please follow the installation step in the [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project.

For image enhancement re-implementation, please follow:
- Create Conda Environment
```bash
conda create -n TripleA python=3.8
conda activate TripleA
```
- Install Dependencies
```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install addict future lmdb numpy opencv-python Pillow pyyaml requests scikit-image scipy tb-nightly tqdm yapf
python setup.py develop
```

## 3. Data preparation
### 3.1 KAIST Multispectral Pedestrian Detection Benchmark
Download the KAIST training and validation images from [here](https://github.com/SoonminHwang/rgbt-ped-detection/zipball/master).

### 3.2 The EuroCity Persons Dataset
Download the ECP training and validation images from the official [website](https://eurocity-dataset.tudelft.nl/).

## 4. Training
- Distribution Alignment
```bash
conda activate cyclegan
python train.py --dataroot ./datasets/ECP --model cycle_gan --lambda_cycedge 1.0 --batch_size 8
```

## 5. Testing
Please download our trained models, and put them in folder ```pretrained_weights```.
