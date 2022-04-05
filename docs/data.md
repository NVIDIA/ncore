## Image metadata
Image metadata is saved in `****.pkl` files, which can be read as:

```
with open(`****.pkl`,'rb') as file:
    img_metadata = pickle.load(file)
```

image metadata then contains the following data:

```

```


## Lidar data


## Poses


## Autolabels

This document describes the data formats and APIs. Graphical overview of the data structure and processing pipeline is available [here](https://docs.google.com/presentation/d/1JGDH8g2PiUIWdcu3nmLBIJLt_I7dOJhlbH-QTKrz1cg/edit?usp=sharing).

### Raw data

Raw data is obtained from various sources:
- DeepMap - Poses, Lidar, Calibration, Timestamps
- Maglev - Videos of all cameras
- Jonathan Howe - 4D autolabels

This raw data can be converted into our default format using the `script/convert_raw_data.py` script, which expects the following data structure: 
```
session_name
│   calibrated_rig.json # Rig calibration file
│   aligned_track_records.pb.txt # Ego motion and timestamps
│   to_vehicle_transform_lidar00.pb.txt # Lidar to rig trans.
│
└───cameras
│   │   camera_front_wide_120fov.mp4
│   │   camera_front_fisheye_200fov.mp4
│   │   ...
│   │   camera_front_wide_120fov.mp4.timestamps
│   │   camera_front_fisheye_200fov.mp4.timestamps
│   │   ...
│
└───lidar_00
│   │   ***.pb # Individual lidar spin 
│   │   ...
│ 
└───labels
    │  autolabels.parquet # Autolabel data
```

The script will export the data in the following folder structure, which contains individual camera frames with metadata, semantic segmentation and dynamic masks, unwinded point clouds, poses, and 4D autolabels. Specific formats are described [here](data_formats.md).

```
session_name
└───images # 00-05 = wide angle, 10-13 = fisheye cameras
│   │   timestamps.npz # Timestamp of each frame
│   └───image_00 
│   │   │   0000.jpg         # Single frame
│   │   │   0000.pkl         # Image metadata (see below)
│   │   │   sem_seg_0000.png # Semantic segmentation
│   │   │   dynamic_mask.png # Dynamic mask
│   │   │   ...
│   └───...
│   └───image_05
│   └───image_10
│   └───...
│   └───image_13
│
└───lidar 
│   │   timestamps.npz # Timestamp of each frame
│   │   0000.ply # Single lidar spin nx3
│   │   0000.dat # 3D rays in space nx9 (see below)
│   │   ...
│
└───labels_4d
│   │   autolabels.pkl # 4D autolabels (see below)
└───poses
│   │   poses.npz # Global poses and corresponding timestamps
│   │   rig.npz   # Rig transformations (e.g. lidar2rig)
```