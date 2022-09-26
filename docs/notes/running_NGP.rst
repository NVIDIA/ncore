Running NGP on the DriveSimAI data
==================================

To reconstruct the radiance field of the static background we use the Instant-NGP. 

Compiling Instant-NGP 
---------------------

To compile Instant-NGP start by cloning the repo from the [internal repository](https://gitlab-master.nvidia.com/tmueller/neural-graphics-primitives) by running::


    git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/tmueller/neural-graphics-primitives.git
    cd neural-graphics-primitives


FAQ, compiliation problems, and other usefull information are available in the [public repository](https://github.com/NVlabs/instant-ngp).

Then, use CMake to build the project::


    neural-graphics-primitives$ cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/{cuda-version}/bin/nvcc   
    neural-graphics-primitives$ cmake --build build --config RelWithDebInfo -j 16


Generating config files
-----------------------

Instant-NGP uses `*.json` config files to initialize the parameters and image/lidar paths. The config files for DriveSimAi data can be generated using the `dsai_to_ngp.py` script. For example, the command::


    python scripts/dsai_to_ngp.py --root-dir /path/to/data/ --experiment-name dummy_experiment --start-frame 0 --end-frame 200 
                                            --step-frame 2 -c 0 -c 1 --max-dist 200 --use-lidar nvidia


will generate config files for cameras (`-c`) `0` and `1` and save them in `/path/to/data/ngp_configs/dummy_experiment`. Runing Instan-NGP using the generated config files will fir a model using every ***third*** frame between ***0th*** and ***200th*** frame. Lidar data (`--use-lidar`) will also be included. Due to the memory constraints, currently only ~200 images of (4k x 2k - Nvidia resolution) can be used. 

Running Instant-NGP
-------------------

Give the configs file generated above, Instant-NGP can be run as follows::


    cd instant-ngp
    python ./scripts/run.py --mode nerf --train --scene /path/to/data/ngp_configs/dummy_experiment/camera_0.json  --n_steps 20000  --width 3848 --height 2168 --save_snapshot ./pretrained_models/dummy_experiment.msgpack --near_distance 0.0 --gui


This will train Instant-NGP for 20000 iterrations before saving the pretrained weights to `./pretrained_models/dummy_experiment.msgpack`. For more information on individual CL parameters please check the original Instant-NGP repository.

