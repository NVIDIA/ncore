# DriveSim-AI

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC <zgojcic@nvidia.com>/OR LITANY <olitany@nvidia.com>.

NOTE: This codebase is under active development and the APIs may thus still change. If you build upon this repository, consider forking it to prevent such issues.


# Installation 

## Create a virtual environment 

```
conda create -n drivesim_ai python=3.8
conda activate drivesim_ai
pip install --upgrade pip
pip install waymo-open-dataset-tf-2-5-0 --user
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
pip install scipy
pip install runx
pip install setuptools==59.5.0
pip install opencv-python
pip install scikit-image
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install apex as 

```
cd dependencies
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../..
```
## Compile all the proto files 

Protofiles can be compiled using:

```
protoc --python_out=. protos/*.proto
```

If you do not have the compiler installed on your system please download it from [here](https://developers.google.com/protocol-buffers/docs/downloads).

## Download the pretrained weights 

### Semantic-segmentation

Download the `cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth` and `hrnetv2_w48_imagenet_pretrained.pth` models from [here])(https://drive.google.com/drive/folders/1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) and place them in the `dependencies/semantic-segmentation/pretrained_models/`.

### Instance-segmentation

Download the `model_final_ba17b9.pkl` model from [here])(https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl) and place it in the `dependencies/instance-segmentation/pretrained_models/`.
