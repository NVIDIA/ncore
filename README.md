# DriveSim-AI

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC <zgojcic@nvidia.com>/OR LITANY <olitany@nvidia.com>.

NOTE: This codebase is under active development and the APIs may thus still change. If you build upon this repository, consider forking it to prevent such issues.

# Installation 

## Install git-lfs

```
sudo apt-get install git-lfs
git lfs install
```
[one-time operation]

## Clone repo with submodules

```
git clone --recursive https://gitlab-master.nvidia.com/zgojcic/drivesim-ai.git
```

## Create a virtual environment 

```
cd drivesim-ai
conda create -n drivesim_ai python=3.8
conda activate drivesim_ai
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
sudo apt-get install python3-tk

```

Build the `av_utils` package with: 
```angular2html
cd lib/
python setup.py develop
cd ..
```

Install `apex` as 

```
cd dependencies/apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../..
```

## Compile all the `.proto` files 

Protofiles can be compiled using:

```
protoc --python_out=. protos/*.proto
```

If you do not have the compiler installed on your system please download it from [here](https://developers.google.com/protocol-buffers/docs/downloads).

## Compile the Poisson surface reconstruction

```
sudo apt-get install libpng-dev libjpeg-turbo8-dev
cd dependencies/surface_reconstruction/PoissonRecon
make -j 8
cd ../../..
```
 
## Download the pre-trained weights 

### Semantic-segmentation

Download the `cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth` and `hrnetv2_w48_imagenet_pretrained.pth` models from [here])(https://drive.google.com/drive/folders/1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) and place them in the `dependencies/semantic_segmentation/pretrained_models/`.

### Instance-segmentation

Download the `model_final_ba17b9.pkl` model from [here])(https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl) and place it in the `dependencies/instance-segmentation/pretrained_models/`.
