# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging
import math
import copy
import json
import os

from pathlib import Path
from collections import defaultdict
from typing import Optional
from PIL import Image

import click
import numpy as np
import scipy
import torch

from ncore.impl.data.data3 import (
    ShardDataLoader,
    CameraSensor,
    LidarSensor
)
from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    FrameTimepoint,
    PinholeCameraModelParameters,
)
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.data.util import padded_index_string
from ncore.impl.common.common import average_camera_pose, save_pc_dat, PoseInterpolator, get_3d_bbox_coords
from ncore.impl.common.transformations import transform_point_cloud, bbox_pose, pose_bbox, transform_bbox
from ncore.impl.common.nvidia_utils import LabelProcessor as NVLabelProcessor

# Rotation of NCORE camera frame to NGP camera frame
R_NCORE_NGP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# Encodes the rolling shutter parameters [a,b,c] such y = a + b*x + c*y (Waymo rolling shutter is column wise)
RS_DIR_TO_NGP = {
    'ROLLING_TOP_TO_BOTTOM': np.array([0.0, 0.0, 1.0]),
    'ROLLING_LEFT_TO_RIGHT': np.array([0.0, 1.0, 0.0]),
    'ROLLING_BOTTOM_TO_TOP': np.array([1.0, 0.0, -1.0]),
    'ROLLING_RIGHT_TO_LEFT': np.array([1.0, -1.0, 0.0])
}


def extract_dynamic_tracks(lidar_sensor: LidarSensor, lidar_frame_range: range, track_speed_thresh: float,
                           track_unconditionally_dynamic_classes: set[str]) -> dict[str, dict]:

    ## Extract the dynamic tracks for the given data range
    assert track_speed_thresh >= 0.0, "Speed threshold for autolabeling tracks needs to be a positive value"

    all_tracks: dict[str, dict] = {}

    # Extend the lidar frame range so that we cover all the images
    extended_lidar_frame_range = range(max(lidar_frame_range.start - 1, 0),
                                       min(lidar_frame_range.stop + 1,
                                           lidar_sensor.get_frame_index_range()[-1]))

    # Iterate over all lidar frames and extract ALL tracks
    for frame_idx in extended_lidar_frame_range:
        T_sensor_world = lidar_sensor.get_frame_T_sensor_world(frame_idx)
        labels = lidar_sensor.get_frame_labels(frame_idx)
        for label in labels:
            if label.track_id in all_tracks:
                # Extend existing track
                all_tracks[label.track_id]['max_global_speed'] = max(all_tracks[label.track_id]['max_global_speed'], label.global_speed)
                all_tracks[label.track_id]['poses'].append(bbox_pose(transform_bbox(label.bbox3.to_array(), T_sensor_world)))
                all_tracks[label.track_id]['timestamps_us'].append(label.timestamp_us)
            else:
                # Instantiate new track
                all_tracks[label.track_id] = {
                    'unconditionally_dynamic': label.label_class in track_unconditionally_dynamic_classes, # Some objects are unconditionally dynamic
                    'max_global_speed': label.global_speed,
                    'poses': [bbox_pose(transform_bbox(label.bbox3.to_array(), T_sensor_world))],
                    'dimension': label.bbox3.to_array()[3:6],
                    'label_class': label.label_class,
                    'timestamps_us': [label.timestamp_us]
                }

    # Extract ONLY the dynamic trajectories based on the speed threshold
    dynamic_tracks: dict[str, dict] = {}
    for track_id, track in all_tracks.items():
        if (track['max_global_speed'] > track_speed_thresh or track['unconditionally_dynamic']) and len(track['timestamps_us']) > 1:
            dynamic_tracks[track_id] = track
            dynamic_tracks[track_id]['pose_interpolator'] = PoseInterpolator(np.stack(track['poses']), track['timestamps_us'])

    return dynamic_tracks

@click.command()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
@click.option('--output-dir', type=str, help='Path to the output folder', required=True)
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
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, front wide camera if not specified)",
    default=["camera_front_wide_120fov"],
)
@click.option(
    "--max-dist",
    type=float,
    help="Maximum distance from each camera pose",
    default=150.0,
)
@click.option(
    "--static-camera-mask-dilations",
    type=int,
    help="Number of image dilations to apply to static camera masks",
    default=20,
)
@click.option("--aabb-scale", type=float, help="The desired aabb scale", default=16.0)
@click.option(
    "--lidar-id",
    "lidar_id",
    default=None,
    type=str,
    help="If provided, the lidar sensor to incorporate point clouds from",
)
@click.option(
    "--track-speed-thresh",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Speed threshold for cuboid tracks to be considered dynamic [m/s]",
    default=1.5,
)
@click.option(
    "--track-mask-dilate-ratio",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Ratio for cuboid -> dynamic mask image projection (1.0 results in no dilation, values smaller than 1.0 shrink the mask)",
    default=1.4,
)
@click.option(
    "--track-unconditionally-dynamic-class",
    "track_unconditionally_dynamic_classes",
    multiple=True,
    type=str,
    help="Label classes to treat as unconditionally dynamic",
    default=list(NVLabelProcessor.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC),
)
@click.option(
    "--track-ftheta-max-fov-deg",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Limit FOV for FTheta camera to this range to prevent erronous track projections (in particular for large-FOV fisheye cameras)",
    default=190.0,
)
@click.option(
    "--save-test",
    is_flag=True,
    default=False,
    help="Save the test configs with the same parameters as train",
)
def ncore_to_ngp(
    shard_file_pattern: str,
    output_dir: str,
    experiment_name: str,
    start_frame: int,
    end_frame: int,
    step_frame: int,
    camera_ids: list[str],
    max_dist: float,
    static_camera_mask_dilations: int,
    aabb_scale: float,
    lidar_id: Optional[str],
    track_speed_thresh: float,
    track_mask_dilate_ratio: float,
    track_unconditionally_dynamic_classes: list[str],
    track_ftheta_max_fov_deg: float,
    save_test: bool,
):
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)

    # Create output paths
    output_path_data = Path(output_dir) / loader.get_sequence_id(with_shard_range=True)
    output_path_data.mkdir(parents=True, exist_ok=True)
    assert not (output_path_experiment := output_path_data / 'ngp_configs' / experiment_name
               ).exists(), "Experiment with the same name already exists"
    output_path_experiment.mkdir(parents=True)

    logger.info(f"Preparing NGP config for experiment '{experiment_name}' in '{output_path_experiment}'")

    assert len(camera_ids), "Require at least a single camera sensor"

    # Load lidar time-range and labels
    dynamic_tracks: dict[str, dict]= {}
    if lidar_id:
        logger.info(f"Preparing dynamic objects from '{lidar_id}'")

        # Load sensors
        assert isinstance(reference_camera_sensor := loader.get_sensor(camera_ids[0]), CameraSensor)
        assert isinstance(lidar_sensor := loader.get_sensor(lidar_id), LidarSensor)

        # Find the corresponding lidar frames based on their timestamps
        reference_camera_timestamps = reference_camera_sensor.get_frames_timestamps_us()
        lidar_timestamps = lidar_sensor.get_frames_timestamps_us()
        lidar_frame_start_idx = (np.where(lidar_timestamps > reference_camera_timestamps[start_frame])[0][0] + 1)
        lidar_frame_end_idx = (np.where(lidar_timestamps < reference_camera_timestamps[end_frame])[0][-1] + 1
                               )  # add lidar at the end as cameras see further away

        dynamic_tracks = extract_dynamic_tracks(
            lidar_sensor=lidar_sensor,
            lidar_frame_range=range(lidar_frame_start_idx, lidar_frame_end_idx),
            track_speed_thresh=track_speed_thresh,
            track_unconditionally_dynamic_classes=set(track_unconditionally_dynamic_classes))

    all_camera_data: dict[str, dict] = defaultdict(dict)
    for camera_id in camera_ids:
        logger.info(f"Processing camera '{camera_id}'")

        assert isinstance(camera_sensor := loader.get_sensor(camera_id), CameraSensor)

        camera_data = all_camera_data[camera_id]

        # Get camera intrinsic properties compatible with NGP
        camera_model_parameters = camera_sensor.get_camera_model_parameters()
        match camera_model_parameters:
            case FThetaCameraModelParameters(
                resolution=resolution,
                principal_point=principal_point,
                reference_poly=reference_poly,
                pixeldist_to_angle_poly=bw_poly,
                angle_to_pixeldist_poly=fw_poly,
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
                    "ftheta_f0": float(fw_poly[0]),
                    "ftheta_f1": float(fw_poly[1]),
                    "ftheta_f2": float(fw_poly[2]),
                    "ftheta_f3": float(fw_poly[3]),
                    "ftheta_f4": float(fw_poly[4]),
                    "ftheta_f5": float(fw_poly[5]),
                    "reference_poly": reference_poly.name,
                    "rolling_shutter": rolling_shutter.tolist(),
                    "camera_angle_x": fov_angle_x,
                }

                # Restrict effective FOV for cuboid projections to prevent issues with *invalid* points projected back into the valid image domain FOV
                # - this way they get properly classified as invalid
                camera_model_parameters.max_angle = min(np.deg2rad(track_ftheta_max_fov_deg) / 2.0, camera_model_parameters.max_angle)

            case PinholeCameraModelParameters() as pinhole:
                # Get the focal length and compute the angular field of view
                fov_angle_x = math.atan(pinhole.resolution[0]/(pinhole.focal_length[0]*2))*2
                fov_angle_y = math.atan(pinhole.resolution[1]/(pinhole.focal_length[1]*2))*2

                # Extract the rolling shutter parameters representing y = a + b*x + c*y
                rolling_shutter = RS_DIR_TO_NGP[pinhole.shutter_type.name]

                if not np.isclose(pinhole.radial_coeffs[2], 0).all():
                    logger.warn(f'Pinhole camera model of {camera_id} has non-zero radial distortion coefficient k3, '
                                 'which might not be supported by NGP yet - exporting anyway')

                if not np.isclose(pinhole.radial_coeffs[3:], 0).all():
                    logger.warn(f'Pinhole camera model of {camera_id} has non-zero rational radial distortion coefficients [k4,k5,k6], '
                                 'which might not be supported by NGP yet - exporting anyway')

                if not np.isclose(pinhole.thin_prism_coeffs, 0).all():
                    logger.warn(f'Pinhole camera model of {camera_id} has non-zero thin-prism distortion coefficients [s1,s2,s3,s4], '
                                 'which might not be supported by NGP yet - exporting anyway')

                camera_data["intrinsic_data"] = {
                    "w": int(pinhole.resolution[0]),
                    "h": int(pinhole.resolution[1]),
                    "cx": float(pinhole.principal_point[0]),
                    "cy": float(pinhole.principal_point[1]),
                    "k1": float(pinhole.radial_coeffs[0]),
                    "k2": float(pinhole.radial_coeffs[1]),
                    "p1": float(pinhole.tangential_coeffs[0]),
                    "p2": float(pinhole.tangential_coeffs[1]),
                    # Note: we are already outputting these higher-order / rational radial distortion and thin-prism coefficients, although NGP might not use it yet
                    "k3": float(pinhole.radial_coeffs[2]),
                    "k4": float(pinhole.radial_coeffs[3]),
                    "k5": float(pinhole.radial_coeffs[4]),
                    "k6": float(pinhole.radial_coeffs[5]),
                    "s1": float(pinhole.thin_prism_coeffs[0]),
                    "s2": float(pinhole.thin_prism_coeffs[1]),
                    "s3": float(pinhole.thin_prism_coeffs[2]),
                    "s4": float(pinhole.thin_prism_coeffs[3]),
                    "rolling_shutter": rolling_shutter.tolist(),
                    "camera_angle_x": fov_angle_x,
                    "camera_angle_y": fov_angle_y
                }
            case _:
                raise TypeError(
                    f"unsupported camera model type {type(camera_model_parameters)}, currently supporting Ftheta/Pinhole only"
                )

        camera_model = CameraModel.from_parameters(camera_model_parameters,
                                                   device='cpu' # we only project small number of points
                                                   )

        # Z is the up vector in the NCORE coordinate system
        camera_data["up"] = np.array([0, 0, 1])

        def nvidia_to_ngp(T_sensor_to_world):
            R = T_sensor_to_world[:3, :3] @ R_NCORE_NGP
            T_sensor_to_world[:3, :3] = R
            return T_sensor_to_world

        # Prepare static component of dynamic masks
        if camera_mask_image := camera_sensor.get_camera_mask_image():
            # Apply fixed number of dilations to static input
            ego_car_mask = scipy.ndimage.binary_dilation(np.asarray(camera_mask_image) != 0, iterations=static_camera_mask_dilations).astype(np.uint8)
            # Force all non-zero values to 255
            ego_car_mask = np.where(ego_car_mask == 0, 0, 255).astype(np.uint8)
        else:
            # Initialize as empty
            ego_car_mask = np.zeros((camera_data['intrinsic_data']['h'], camera_data['intrinsic_data']['w']), dtype=np.uint8)

        # Prepare all image paths (to average the pose we neglect the selected step size as all images will be used for testing) and poses
        all_image_paths: list[Path] = []
        T_cam_rig: list[np.ndarray] = []
        poses_start: list[np.ndarray] = []
        poses_end: list[np.ndarray] = []
        for camera_frame_idx in camera_sensor.get_frame_index_range(start_frame, end_frame):
            camera_path = output_path_data / 'cameras' / camera_sensor.get_sensor_id()
            camera_path.mkdir(parents=True, exist_ok=True)

            # check if camera image data was already exported / export it otherwise
            IMAGE_FORMAT='jpeg'
            if not (frame_file_path := camera_path / Path(padded_index_string(camera_frame_idx)).with_suffix(f'.{IMAGE_FORMAT}')).exists():
                frame_data_handle = camera_sensor.get_frame_data(camera_frame_idx)
                frame_data_format = frame_data_handle.get_encoded_image_format()
                assert(frame_data_format == IMAGE_FORMAT), f"update conversion script to support encoding '{frame_data_format}' images"
                with open(frame_file_path, "wb") as frame_file:
                    frame_file.write(frame_data_handle.get_encoded_image_data())

            all_image_paths.append(frame_file_path)

            T_cam_rig.append(camera_sensor.get_T_sensor_rig())
            poses_start.append(nvidia_to_ngp(camera_sensor.get_frame_T_sensor_world(camera_frame_idx, FrameTimepoint.START)))
            poses_end.append(nvidia_to_ngp(camera_sensor.get_frame_T_sensor_world(camera_frame_idx, FrameTimepoint.END)))

            # extend static mask with dynamic parts
            if not (mask_file_path := frame_file_path.parent / ("dynamic_mask_" + frame_file_path.stem + '.png')).exists():
                dynamic_mask = ego_car_mask.copy()

                # check which dynamic labels where observed at this frame's time + project
                frame_start_timestamp_us = camera_sensor.get_frame_timestamp_us(camera_frame_idx, FrameTimepoint.START)
                frame_end_timestamp_us = camera_sensor.get_frame_timestamp_us(camera_frame_idx, FrameTimepoint.END)
                frame_mid_timestamp = frame_start_timestamp_us + (frame_end_timestamp_us - frame_start_timestamp_us) // 2
                for dynamic_track in dynamic_tracks.values():
                    if not (dynamic_track['timestamps_us'][0] <= frame_mid_timestamp and frame_mid_timestamp <= dynamic_track['timestamps_us'][-1]):
                        continue

                    # interpolate track to frame mid-timestamp
                    bbox_pose = dynamic_track['pose_interpolator'].interpolate_to_timestamps(frame_mid_timestamp)[0]

                    bbox = pose_bbox(bbox_pose, dynamic_track['dimension'])

                    bbox_corners = get_3d_bbox_coords(bbox)

                    projection = camera_model.world_points_to_image_points_shutter_pose(
                            bbox_corners,
                            camera_sensor.get_frame_T_world_sensor(camera_frame_idx, FrameTimepoint.START),
                            camera_sensor.get_frame_T_world_sensor(camera_frame_idx, FrameTimepoint.END),
                            return_valid_indices=True, return_T_world_sensors=True, return_all_projections=True)

                    if torch.numel(projection.valid_indices) > 0:
                        # Clamped out-of-image-domain points
                        #   - this is required for perspective cameras only
                        #   - fisheye-cameras will project in a "clamped way" into the image-domain along
                        #     it's internal FOV-specific bounds, but points will be marked as invalid
                        projection.image_points[:,0] = torch.clamp(projection.image_points[:,0], min=0, max=camera_model.resolution[0])
                        projection.image_points[:,1] = torch.clamp(projection.image_points[:,1], min=0, max=camera_model.resolution[1]) 

                        min_x, min_y = projection.image_points.min(0)[0]
                        max_x, max_y = projection.image_points.max(0)[0]

                        mask_width_padding = torch.ceil((max_x - min_x) * (track_mask_dilate_ratio - 1.0) / 2).to(torch.int32)
                        mask_height_padding =  torch.ceil((max_y - min_y) * (track_mask_dilate_ratio - 1.0) / 2).to(torch.int32)

                        max_x_int, max_y_int = torch.ceil(max_x).to(torch.int32) + mask_width_padding, torch.ceil(max_y).to(torch.int32) + mask_height_padding
                        min_x_int, min_y_int = torch.floor(min_x).to(torch.int32) - mask_width_padding, torch.floor(min_y).to(torch.int32) - mask_height_padding

                        min_x = torch.clamp(min_x_int, min=0, max=camera_model.resolution[0])
                        min_y = torch.clamp(min_y_int, min=0, max=camera_model.resolution[1])
                        max_x = torch.clamp(max_x_int, min=0, max=camera_model.resolution[0])
                        max_y = torch.clamp(max_y_int, min=0, max=camera_model.resolution[1])

                        dynamic_mask[min_y:max_y, min_x:max_x] = 255

                Image.fromarray(dynamic_mask).save(mask_file_path, bits=1, optimize=True)
                
        # Train images respect step size
        all_image_train_paths: list[Path] = all_image_paths[::step_frame]

        poses_start = np.stack(poses_start)
        poses_end = np.stack(poses_end)

        camera_data["poses_start"] = poses_start
        camera_data["poses_end"] = poses_end

        camera_data["all_image_paths"] = all_image_paths
        camera_data["all_image_train_paths"] = all_image_train_paths

    # Combine all the poses and compute the scaling factor and centroid, use the start timestamp pose as approximation
    all_poses = [all_camera_data[camera_id]["poses_start"] for camera_id in camera_ids]
    all_poses = np.concatenate(all_poses, axis=0)

    pose_avg, extent = average_camera_pose(all_poses)
    scale_factor = 1 / ((extent / 2 + max_dist) / (aabb_scale / 2))  # So that the max far is scaled to the target scale
    offset = -(pose_avg * scale_factor) + np.array([0.5, 0.5, 0.5])  # Instant NGP assumes that the scenes are centered at 0.5^3

    # Generate a config file for each of the cameras
    for camera_sensor_idx, camera_id in enumerate(camera_ids):

        camera_data = all_camera_data[camera_id]

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
                "file_path": os.path.relpath(image_train_path, output_path_experiment),
                "transform_matrix_start": camera_data["poses_start"][step_frame * i].tolist(),
                "transform_matrix_end": camera_data["poses_end"][step_frame * i].tolist(),
            } | intrinsic_data

            out_train["frames"].append(frame_data)

        for i, image_path in enumerate(camera_data["all_image_paths"]):
            frame_data = {
                "file_path": os.path.relpath(image_path, output_path_experiment),
                "transform_matrix_start": camera_data["poses_start"][i].tolist(),
                "transform_matrix_end": camera_data["poses_end"][i].tolist(),
            }

            out_test["frames"].append(frame_data)

        if camera_sensor_idx == 0 and lidar_id:
            logger.info(f"Processing lidar '{lidar_id}'")

            # Store lidar data as '.dat' files as required by NGP
            out_train["lidar"] = []

            lidar_path = output_path_data / 'lidars' / lidar_sensor.get_sensor_id()
            lidar_path.mkdir(parents=True, exist_ok=True)
            for lidar_frame_idx in range(lidar_frame_start_idx, lidar_frame_end_idx):
                if not (dat_path := lidar_path / Path(padded_index_string(lidar_frame_idx)).with_suffix('.dat')).exists():
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

                out_train["lidar"].append({"file_path": os.path.relpath(dat_path, output_path_experiment)})

        train_camera_path = output_path_experiment / f"{camera_id}_train.json"
        logger.info(f"Writing '{train_camera_path}'")
        with open(train_camera_path, "w") as f:
            json.dump(out_train, f, indent=2)

        if save_test:
            test_camera_path = output_path_experiment / f"{camera_id}_test.json"
            logger.info(f"Writing '{test_camera_path}'")
            with open(test_camera_path, "w") as f:
                json.dump(out_test, f, indent=2)


if __name__ == "__main__":
    ncore_to_ngp()
