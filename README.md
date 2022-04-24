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
```

## Compile all the proto files 

Protofiles can be compiled using:

```
protoc --python_out=. protos/*.proto
```

If you do not have the compiler installed on your system please download it from [here](https://developers.google.com/protocol-buffers/docs/downloads).