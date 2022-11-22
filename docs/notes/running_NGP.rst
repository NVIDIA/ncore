.. Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

Running NGP on DSAI data
=============================

To reconstruct the radiance field of the static background we use the Instant-NGP. 

Compiling Instant-NGP 
---------------------

To compile Instant-NGP start by cloning the repo from the [internal repository](https://gitlab-master.nvidia.com/tmueller/neural-graphics-primitives) by running::


    git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/tmueller/neural-graphics-primitives.git
    cd neural-graphics-primitives


FAQ, compilation problems, and other useful information are available in the [public repository](https://github.com/NVlabs/instant-ngp).

Then, use CMake to build the project::


    neural-graphics-primitives$ cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/{cuda-version}/bin/nvcc   
    neural-graphics-primitives$ cmake --build build --config RelWithDebInfo -j 16


Generating config files
-----------------------

Instant-NGP uses `*.json` config files to initialize the parameters and image/lidar paths.
The config files for DSAI data can be generated using the `dsai_to_ngp` target. For example, the command::


    bazel run //scripts:dsai_to_ngp \
        -- \
        --root-dir=<PATH-TO-DATA> \
        --experiment-name=dummy_experiment \
        --camera-sensor=camera_front_wide_120fov \
        --camera-sensor=camera_cross_left_120fov \
        --camera-sensor=camera_cross_right_120fov \
        --lidar-sensor=lidar_gt_top_p128_v4p5 \
        --start-frame 0 \
        --end-frame 200 \
        --step-frame 2 \
        --max-dist 200


will generate config files for three cameras and save them in `<PATH-TO-DATA>/ngp_configs/dummy_experiment`.
Running Instant-NGP using the generated config files will fit a model using every ***third*** frame between ***0th*** and ***200th*** frame.
Lidar data will also be included. Due to the memory constraints, currently only ~200 images of (4k x 2k - Nvidia resolution) should be used. 

Running Instant-NGP
-------------------

Give the configs file generated above, Instant-NGP can be run as follows::


    cd instant-ngp
    python ./scripts/run.py \
        --mode nerf \
        --train \
        --scene <PATH-TO-DATA>/ngp_configs/dummy_experiment/camera_front_wide_120fov_train.json \
        --n_steps 20000 \
        --width 3848 \
        --height 2168 \
        --save_snapshot ./pretrained_models/dummy_experiment.msgpack \
        --near_distance 0.0 \
        --gui


This will train Instant-NGP for 20000 iterations before saving the pretrained weights to `./pretrained_models/dummy_experiment.msgpack`.
For more information on individual CL parameters please check the original Instant-NGP repository.
