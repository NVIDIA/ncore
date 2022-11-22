# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging
import math
import copy
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

import click
import numpy as np

from src.dsai_internal.data.data import (
    DataLoader,
    CameraSensor,
    LidarSensor,
    FThetaCameraModelParameters,
    FrameTimepoint,
    PinholeCameraModelParameters,
)
from src.dsai_internal.data.util import padded_index_string

from src.dsai_internal.common.common import average_camera_pose, save_pc_dat
from src.dsai_internal.common.nvidia_utils import transform_point_cloud

# Rotation of DSAI camera frame to NGP camera frame
R_DSAI_NGP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# Encodes the rolling shutter parameters [a,b,c] such y = a + b*x + c*y (Waymo rolling shutter is column wise)
RS_DIR_TO_NGP = {
    'ROLLING_TOP_TO_BOTTOM': np.array([0.0, 0.0, 1.0]),
    'ROLLING_LEFT_TO_RIGHT': np.array([0.0, 1.0, 0.0]),
    'ROLLING_BOTTOM_TO_TOP': np.array([1.0, 0.0, -1.0]),
    'ROLLING_RIGHT_TO_LEFT': np.array([1.0, -1.0, 0.0])
}


@click.command()
@click.option("--root-dir", type=str, help="Path to the preprocessed sequence", required=True)
@click.option("--experiment-name", type=str, help="Name of the experiment", required=True)
@click.option(
    "--start-frame",
    type=click.IntRange(min=0, max_open=True),
    help="Initial camera frame to be use",
    default=0,
)
@click.option(
    "--end-frame",
    type=click.IntRange(min=-1, max_open=True),
    help="End camera frame to be used",
    default=-1,
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of frames",
    default=1,
)
@click.option(
    "--camera-sensor",
    "camera_sensor_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, all if not specified)",
    default=["camera_front_wide_120fov"],
)
@click.option(
    "--max-dist",
    type=float,
    help="Maximum distance from each camera pose.",
    default=150.0,
)
@click.option("--aabb-scale", type=float, help="The desired aabb scale", default=16.0)
@click.option(
    "--lidar-sensor",
    "lidar_sensor_id",
    default=None,
    type=str,
    help="If provided, the lidar sensor to incorporate point clouds from",
)
@click.option(
    "--save-test",
    is_flag=True,
    default=False,
    help="Save the test configs with the same parameters as train",
)
def dsai_to_ngp(
    root_dir: str,
    experiment_name: str,
    start_frame: int,
    end_frame: int,
    step_frame: int,
    camera_sensor_ids: list[str],
    max_dist: float,
    aabb_scale: float,
    lidar_sensor_id: Optional[str],
    save_test: bool,
):
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    loader = DataLoader(root_dir)

    # Create output path
    assert not (output_dir := loader.get_sequence_dir() /
                (f"ngp_configs/{experiment_name}")).exists(), "Experiment with the same name already exists"
    output_dir.mkdir(parents=True)

    logger.info(f"Preparing NGP config in '{output_dir}'")

    all_camera_data: dict[str, dict] = defaultdict(dict)
    for camera_sensor_id in camera_sensor_ids:
        assert isinstance(camera_sensor := loader.get_sensor(camera_sensor_id), CameraSensor)

        camera_data = all_camera_data[camera_sensor_id]

        # Get camera intrinsic properties compatible with NGP
        match camera_model_parameters := camera_sensor.get_camera_model_parameters():
            case FThetaCameraModelParameters(
                resolution=resolution,
                principal_point=principal_point,
                bw_poly=bw_poly,
                fw_poly=fw_poly,
                shutter_type=shutter_type,
            ):
                # This is not really the focal length, only a crude approximation, but NGP expects some value for fov (not used in training)
                focal_length = fw_poly[1]
                fov_angle_x = math.atan2(resolution[0], focal_length * 2) * 2

                # Extract the rolling shutter parameters representing y = a + b*x + c*y
                rolling_shutter = RS_DIR_TO_NGP[shutter_type.name]

                assert len(bw_poly) == 6, "Update polynomial export in case the internal order of distortion polynomials changed"
                camera_data["intrinsic_data"] = {
                    "w": int(resolution[0]),
                    "h": int(resolution[1]),
                    "cx": float(principal_point[0]),
                    "cy": float(principal_point[1]),
                    "ftheta_p0": float(bw_poly[0]),
                    "ftheta_p1": float(bw_poly[1]),
                    "ftheta_p2": float(bw_poly[2]),
                    "ftheta_p3": float(bw_poly[3]),
                    "ftheta_p4": float(bw_poly[4]),
                    # Note: we are already outputting this higher-order coefficient although NGP might not use it yet
                    "ftheta_p5": float(bw_poly[5]),
                    "rolling_shutter": rolling_shutter.tolist(),
                    "camera_angle_x": fov_angle_x,
                }

            case PinholeCameraModelParameters() as pinhole:
                # Get the focal length and compute the angular field of view
                fov_angle_x = math.atan(pinhole.resolution[0]/(pinhole.focal_length_u*2))*2
                fov_angle_y = math.atan(pinhole.resolution[1]/(pinhole.focal_length_v*2))*2

                # Extract the rolling shutter parameters representing y = a + b*x + c*y
                rolling_shutter = RS_DIR_TO_NGP[pinhole.shutter_type.name]

                camera_data["intrinsic_data"] = {
                    "w": int(pinhole.resolution[0]),
                    "h": int(pinhole.resolution[1]),
                    "cx": float(pinhole.principal_point[0]),
                    "cy": float(pinhole.principal_point[1]),
                    "k1": pinhole.k1,
                    "k2": pinhole.k2,
                    # Note: we are already outputting this higher-order coefficient although NGP might not use it yet
                    "k3": pinhole.k3,
                    "p1": pinhole.p1,
                    "p2": pinhole.p2,
                    "rolling_shutter": rolling_shutter.tolist(),
                    "camera_angle_x": fov_angle_x,
                    "camera_angle_y": fov_angle_y
                }
            case _:
                raise TypeError(
                    f"unsupported camera model type {type(camera_model_parameters)}, currently supporting Ftheta/Pinhole only"
                )

        # Z is the up vector in the DSAI coordinate system
        camera_data["up"] = np.array([0, 0, 1])

        def nvidia_to_ngp(T_sensor_to_world):
            R = T_sensor_to_world[:3, :3] @ R_DSAI_NGP
            T_sensor_to_world[:3, :3] = R
            return T_sensor_to_world

        # Collect all image paths (to average the pose we neglect the selected step size as all images will be used for testing) and poses
        all_image_paths: list[Path] = []
        T_cam_rig: list[np.ndarray] = []
        poses_start: list[np.ndarray] = []
        poses_end: list[np.ndarray] = []
        for i in camera_sensor.get_frame_index_range(start_frame, end_frame):
            assert isinstance(frame := camera_sensor.get_frame(i),
                              CameraSensor.FileFrameHandle), "only file-based frames currently supported"

            all_image_paths.append(frame.get_image_file_path())

            T_cam_rig.append(camera_sensor.get_T_sensor_rig())
            poses_start.append(nvidia_to_ngp(camera_sensor.get_frame_T_sensor_world(i, FrameTimepoint.START)))
            poses_end.append(nvidia_to_ngp(camera_sensor.get_frame_T_sensor_world(i, FrameTimepoint.END)))

        # Train images respect step size
        all_image_train_paths: list[Path] = []
        for i in camera_sensor.get_frame_index_range(start_frame, end_frame, step_frame):
            assert isinstance(frame := camera_sensor.get_frame(i),
                              CameraSensor.FileFrameHandle), "only file-based frames currently supported"

            all_image_train_paths.append(frame.get_image_file_path())

        poses_start = np.stack(poses_start)
        poses_end = np.stack(poses_end)

        camera_data["poses_start"] = poses_start
        camera_data["poses_end"] = poses_end

        camera_data["all_image_paths"] = all_image_paths
        camera_data["all_image_train_paths"] = all_image_train_paths

        # Resave all image masks as 'dynamic_mask' symlinks
        if camera_mask_image_path := camera_sensor.get_camera_mask_image_path():
            for image_path in all_image_paths:
                target_camera_mask_image_path = image_path.parent / ("dynamic_mask_" + image_path.stem +
                                                                     camera_mask_image_path.suffix)
                if not (target_camera_mask_image_path.exists() or target_camera_mask_image_path.is_symlink()):
                    target_camera_mask_image_path.symlink_to(camera_mask_image_path)

    # Combine all the poses and compute the scaling factor and centroid, use the start timestamp pose as approximation
    all_poses = [all_camera_data[camera_sensor_id]["poses_start"] for camera_sensor_id in camera_sensor_ids]
    all_poses = np.concatenate(all_poses, axis=0)

    pose_avg, extent = average_camera_pose(all_poses)
    scale_factor = 1 / ((extent / 2 + max_dist) / (aabb_scale / 2))  # So that the max far is scaled to the target scale
    offset = -(pose_avg * scale_factor) + np.array([0.5, 0.5, 0.5])  # Instant NGP assumes that the scenes are centered at 0.5^3

    # Generate a config file for each of the cameras
    for camera_sensor_idx, camera_sensor_id in enumerate(camera_sensor_ids):

        camera_data = all_camera_data[camera_sensor_id]

        intrinsic_data = camera_data["intrinsic_data"]

        out_train = {
            "aabb_scale": aabb_scale,
            "n_extra_learnable_dims": 32,
            "up": camera_data["up"].tolist(),
            "offset": offset.tolist(),
            "scale": scale_factor,
            "max_bound": max_dist,
            "frames": [],
        } | intrinsic_data

        out_test = copy.deepcopy(out_train)

        for i, image_train_path in enumerate(camera_data["all_image_train_paths"]):
            frame_data = {
                "file_path": os.path.relpath(image_train_path, output_dir),
                "transform_matrix_start": camera_data["poses_start"][step_frame * i].tolist(),
                "transform_matrix_end": camera_data["poses_end"][step_frame * i].tolist(),
            } | intrinsic_data

            out_train["frames"].append(frame_data)

        for i, image_path in enumerate(camera_data["all_image_paths"]):
            frame_data = {
                "file_path": os.path.relpath(image_path, output_dir),
                "transform_matrix_start": camera_data["poses_start"][i].tolist(),
                "transform_matrix_end": camera_data["poses_end"][i].tolist(),
            }

            out_test["frames"].append(frame_data)

        if camera_sensor_idx == 0 and lidar_sensor_id:
            logger.info(f"Preparing lidar '{lidar_sensor_id}'")

            # Load sensors
            assert isinstance(camera_sensor := loader.get_sensor(camera_sensor_id), CameraSensor)
            assert isinstance(lidar_sensor := loader.get_sensor(lidar_sensor_id), LidarSensor)

            # Find the corresponding lidar frames based on their timestamps
            camera_timestamps = camera_sensor.get_frames_timestamps_us()
            lidar_timestamps = lidar_sensor.get_frames_timestamps_us()
            lidar_frame_start_idx = (np.where(lidar_timestamps > camera_timestamps[start_frame])[0][0] + 1)
            lidar_frame_end_idx = (np.where(lidar_timestamps < camera_timestamps[end_frame])[0][-1] + 1
                                   )  # add lidar at the end as cameras see further away

            # Store lidar data as '.dat' files as required by NGP
            out_train["lidar"] = []

            for lidar_frame_idx in range(lidar_frame_start_idx, lidar_frame_end_idx):
                dat_path = lidar_sensor.get_sensor_dir() / (padded_index_string(lidar_frame_idx) + ".dat")
                if not dat_path.exists():

                    T_sensor_to_world = lidar_sensor.get_frame_T_sensor_world(lidar_frame_idx).astype(np.float32)

                    # Load relevant frame data for ray structure
                    xyz_s = transform_point_cloud(lidar_sensor.get_frame_data(lidar_frame_idx, "xyz_s"),
                                                  T_sensor_to_world)
                    xyz_e = transform_point_cloud(lidar_sensor.get_frame_data(lidar_frame_idx, "xyz_e"),
                                                  T_sensor_to_world)
                    dist = np.linalg.norm(xyz_s - xyz_e, axis=1)  # N x 1
                    intensity = lidar_sensor.get_frame_data(lidar_frame_idx, "intensity")
                    dynamic_flag = lidar_sensor.get_frame_data(lidar_frame_idx, "dynamic_flag")

                    # Assemble full point-cloud ray structure
                    point_cloud = np.column_stack((xyz_s, xyz_e, dist, intensity, dynamic_flag))

                    # Serialize point cloud
                    save_pc_dat(str(dat_path), point_cloud)

                out_train["lidar"].append({"file_path": os.path.relpath(dat_path, output_dir)})

        train_camera_path = output_dir / f"{camera_sensor_id}_train.json"
        logger.info(f"Writing '{train_camera_path}'")
        with open(train_camera_path, "w") as f:
            json.dump(out_train, f, indent=2)

        if save_test:
            test_camera_path = output_dir / f"{camera_sensor_id}_test.json"
            logger.info(f"Writing '{test_camera_path}'")
            with open(test_camera_path, "w") as f:
                json.dump(out_test, f, indent=2)


if __name__ == "__main__":
    dsai_to_ngp()
