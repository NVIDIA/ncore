# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import logging

import numpy as np

from google.protobuf import text_format
from protobuf_to_dict import protobuf_to_dict

from src.protos.deepmap import track_data_pb2
from src.py.common.nvidia_utils import extract_pose, extract_sensor_2_sdc
from src.py.common.common import save_jsonl
from src.py.dataset_converter import BaseNvidiaDataConverter


@click.command()
@click.option('--input-aligned-track', type=str, help='Path to deepmap aligned-track-record', required=True)
@click.option('--input-lidar-transform', type=str, help='Path to deepmap to-vehicle-transform-lidar', required=True)
@click.option('--output-egomotion', type=str, help='Path to converted PyCSFT egomotion.jsonl', required=True)
def convert_deepmap_egomotion(input_aligned_track: str, input_lidar_transform: str, output_egomotion: str):
    ''' Converts deepmap poses into nvidia-maglev-compatible egomotion.jsonl format '''

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info(f"Converting from '{[input_aligned_track, input_lidar_transform]}' to '{output_egomotion}'")

    ## Parse deepmap poses

    # Extract lidar-to-sdc transformation
    T_lidar_sdc = extract_sensor_2_sdc(input_lidar_transform)
    T_sdc_lidar = np.linalg.inv(T_lidar_sdc)

    # Initialize the track aligned track record structure
    track_data = track_data_pb2.AlignedTrackRecords()

    # Read in the track record data from a proto file
    # This includes camera_records and lidar_records (see track_record proto for more detail)
    with open(input_aligned_track, 'r') as f:
        text_format.Parse(f.read(), track_data)

    # Extract all the lidar paths, timestamps and poses from the track record
    track_data = protobuf_to_dict(track_data)

    # Load all world-poses of DeepMap's SDC frame and convert to world-poses of the lidar frame
    T_lidar_world_poses = []
    lidar_world_pose_timestamps = []

    for frame in track_data['lidar_records'][0]['records']:
        if 'pose' in frame:
            lidar_world_pose_timestamps.append(int(frame['timestamp_microseconds']))

            # Transform world pose of SDC to world pose of lidar
            T_sdc_world = extract_pose(frame['pose'])  # pose of SDC frame in world
            T_lidar_world = T_sdc_world @ T_sdc_lidar  # pose of lidar frame in world
            T_lidar_world_poses.append(T_lidar_world)

    ## Create output
    egomotion_entries = []
    for (T_lidar_world_pose, lidar_world_pose_timestamp) in zip(T_lidar_world_poses, lidar_world_pose_timestamps):
        pose_string: str = ' '.join(
            [np.format_float_scientific(x, unique=True) for x in T_lidar_world_pose.transpose().flatten()])

        # Sanity check: make sure string representation of pose is accurate
        egomotion_pose_reloaded = np.asfarray(pose_string.split(' '), dtype=np.float64).reshape((4, 4)).transpose()

        assert np.linalg.norm(T_lidar_world_pose - egomotion_pose_reloaded) < np.finfo(np.float32).eps

        egomotion_entry = {
            "altitude": None,
            "frame_number": None,
            "gps_speed": None,
            "in_sensor_name_frame": BaseNvidiaDataConverter.LIDAR_SENSORNAME,
            "interpolated": False,
            "latitude": None,
            "longitude": None,
            "pose": pose_string,
            "sensor_name": BaseNvidiaDataConverter.LIDAR_SENSORNAME,
            "timestamp": lidar_world_pose_timestamp,
            "valid": True
        }

        egomotion_entries.append(egomotion_entry)

    save_jsonl(output_egomotion, egomotion_entries)


if __name__ == "__main__":
    convert_deepmap_egomotion()
