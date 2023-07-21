# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import click

import numpy as np

from ncore.impl.common.common import time_bounds, HalfClosedInterval
from ncore.impl.data.data3 import ShardDataLoader, ContainerDataWriter
from ncore.impl.data.types import Poses, FrameTimepoint, FrameLabel3, Tracks, TrackLabel, TrackProperties

from ncore.impl.common.nvidia_utils import LabelProcessor as NVLabelProcessor
from ncore.impl.data_converter.waymo3 import WaymoConverter

@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    ''' Parameters passed to non-command-based CLI part '''
    shard_file_pattern: str
    output_dir: str
    output_basename: Optional[str]
    open_consolidated: bool
    store_shard_meta: bool
    lidar_dynamic_flag_bbox_padding_meters: float
    global_speed_dynamic_threshold: float
    debug: bool


class SubrangeDataWriter:
    ''' Performs data subrange selection and outputs a new container with subselected data '''
    @staticmethod
    def process(
            # Source data
            loader: ShardDataLoader,
            start_timestamp_us: int,
            end_timestamp_us: int,

            # Dynamic-flag processing parameters
            lidar_dynamic_flag_bbox_padding_meters: float,
            global_speed_dynamic_threshold: float,

            # Output specification
            output_dir_path: Path,
            container_name: str,

            # Sensor selection
            camera_ids: Optional[list[str]],
            lidar_ids: Optional[list[str]],
            radar_ids: Optional[list[str]],  # exports all sensors of a give type of not restricted

            # Meta
            store_shard_meta: bool) -> None:

        if camera_ids is None:
            camera_ids = loader.get_camera_ids()
        if lidar_ids is None:
            lidar_ids = loader.get_lidar_ids()
        if radar_ids is None:
            radar_ids = loader.get_radar_ids()

        assert start_timestamp_us < end_timestamp_us, "invalid time bounds"
        subrange_interval_us = HalfClosedInterval(start_timestamp_us, end_timestamp_us + 1)  # make sure to include end-timestamp in interval

        # ContainerDataWriter for all outputs (always single-shard)
        data_writer = ContainerDataWriter(
            output_dir_path,
            container_name,
            camera_ids,
            lidar_ids,
            radar_ids,
            loader.get_calibration_type(),
            loader.get_egomotion_type(),
            loader.get_sequence_id(),
            # always single-shard
            0,
            1,
            store_shard_meta)

        ## Process poses
        source_poses = loader.get_poses()

        # subselect poses
        target_poses_range = subrange_interval_us.cover_range(source_poses.T_rig_world_timestamps_us)
        target_poses = Poses(source_poses.T_rig_world_base, source_poses.T_rig_worlds[target_poses_range],
                             source_poses.T_rig_world_timestamps_us[target_poses_range])

        data_writer.store_poses(target_poses)

        ## Process cameras
        for camera_id in camera_ids:
            camera_sensor = loader.get_camera_sensor(camera_id)

            # subselect frames
            source_frame_timestamps_us = camera_sensor.get_frames_timestamps_us()
            target_frames_range = subrange_interval_us.cover_range(source_frame_timestamps_us)

            # store sensor meta
            data_writer.store_camera_meta(camera_id,
                                          source_frame_timestamps_us[target_frames_range],
                                          camera_sensor.get_T_sensor_rig(),
                                          camera_sensor.get_camera_model_parameters(),
                                          camera_sensor.get_camera_mask_image())

            # store subselected frames
            for subrange_frame_index, source_frame_idx in enumerate(target_frames_range):
                T_rig_worlds = np.stack((camera_sensor.get_frame_T_rig_world(source_frame_idx, FrameTimepoint.START),
                                         camera_sensor.get_frame_T_rig_world(source_frame_idx, FrameTimepoint.END)))
                timestamps_us = np.stack((camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                                          camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END)))
                encoded_image_data = camera_sensor.get_frame_handle(source_frame_idx).get_data()
                data_writer.store_camera_frame(camera_id, subrange_frame_index,
                                               encoded_image_data.get_encoded_image_data(),
                                               encoded_image_data.get_encoded_image_format(), T_rig_worlds,
                                               timestamps_us)

        ## Process lidars
        _, source_track_properties = loader.get_tracks()
        if not source_track_properties:
            logging.warn('No data-specific track-properties available, using "default" track properties')

            # initialize "default" set of unconditional label classes from various datasets
            source_track_properties = TrackProperties(label_ids_unconditionally_dynamic = NVLabelProcessor.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC | WaymoConverter.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC,
                                                      label_ids_unconditionally_static = NVLabelProcessor.LABEL_STRINGS_UNCONDITIONALLY_STATIC | WaymoConverter.LABEL_STRINGS_UNCONDITIONALLY_STATIC)

        # Iterate once over all frames to collect surviving tracks / frame labels
        target_track_labels: dict[str, TrackLabel] = {}
        target_frame_labels: dict[str, dict[int, list[FrameLabel3]]] = {}
        for lidar_id in lidar_ids:
            lidar_sensor = loader.get_lidar_sensor(lidar_id)

            if lidar_id not in target_frame_labels:
                target_frame_labels[lidar_id] = {}

            # subselect frames
            source_frame_timestamps_us = lidar_sensor.get_frames_timestamps_us()
            target_frames_range = subrange_interval_us.cover_range(source_frame_timestamps_us)

            # extract labels from subselected frames
            for source_frame_idx in target_frames_range:

                source_frame_timestamp_us = int(source_frame_timestamps_us[source_frame_idx])

                if source_frame_timestamp_us not in target_frame_labels[lidar_id]:
                    target_frame_labels[lidar_id][source_frame_timestamp_us] = []

                for frame_label in lidar_sensor.get_frame_labels(source_frame_idx):
                    target_frame_labels[lidar_id][source_frame_timestamp_us].append(frame_label)

                    track_id = frame_label.track_id

                    if track_id not in target_track_labels:
                        target_track_labels[track_id] = TrackLabel(sensors = {})

                    if lidar_id not in target_track_labels[track_id].sensors:
                        target_track_labels[track_id].sensors[lidar_id] = []

                    target_track_labels[track_id].sensors[lidar_id].append(source_frame_timestamp_us)

        # Second iteration: store frames
        for lidar_id in lidar_ids:
            lidar_sensor = loader.get_lidar_sensor(lidar_id)

            # subselect frames
            source_frame_timestamps_us = lidar_sensor.get_frames_timestamps_us()
            target_frames_range = subrange_interval_us.cover_range(source_frame_timestamps_us)

            # store sensor meta
            data_writer.store_lidar_meta(lidar_id, source_frame_timestamps_us[target_frames_range], lidar_sensor.get_T_sensor_rig())

            # store subselected frames
            for subrange_frame_index, source_frame_idx in enumerate(target_frames_range):
                T_rig_worlds = np.stack((lidar_sensor.get_frame_T_rig_world(source_frame_idx, FrameTimepoint.START),
                                         lidar_sensor.get_frame_T_rig_world(source_frame_idx, FrameTimepoint.END)))
                timestamps_us = np.stack((lidar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                                          lidar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END)))

                # re-estimate dynamic flags based on local track data
                xyz_e = lidar_sensor.get_frame_data(source_frame_idx, 'xyz_e')
                dynamic_flag, frame_labels = NVLabelProcessor.lidar_dynamic_flag(
                    lidar_id,
                    xyz_e,
                    timestamps_us[1],
                    target_track_labels,
                    target_frame_labels,
                    source_track_properties.label_ids_unconditionally_dynamic,
                    source_track_properties.label_ids_unconditionally_static,
                    lidar_dynamic_flag_bbox_padding_meters,
                    global_speed_dynamic_threshold)

                semantic_class = lidar_sensor.get_frame_data(
                    source_frame_idx, 'semantic_class') if lidar_sensor.has_frame_data(source_frame_idx, 'semantic_class') else None
                data_writer.store_lidar_frame(lidar_id,
                                              subrange_frame_index,
                                              lidar_sensor.get_frame_data(source_frame_idx, 'xyz_s'),
                                              xyz_e,
                                              lidar_sensor.get_frame_data(source_frame_idx, 'intensity'),
                                              lidar_sensor.get_frame_data(source_frame_idx, 'timestamp_us'),
                                              dynamic_flag,
                                              semantic_class,
                                              frame_labels,
                                              T_rig_worlds,
                                              timestamps_us
                                              )

        data_writer.store_tracks(
            tracks=Tracks(track_labels=target_track_labels),
            # Reuse source-track properties
            track_properties=source_track_properties)

        ## Finalize output
        data_writer.finalize()


@click.group()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
@click.option('--output-dir', type=str, help='Path to the output folder', required=True)
@click.option(
    '--output-basename',
    type=str,
    default=None,
    help=
    'Basename of the generated file - <sequence-id>_<start-time-us>_<end-time-us> will be used by default if not provided',
    required=False)
@click.option('--open-consolidated/--no-open-consolidated', default=True, help='Pre-load shard meta-data?')
@click.option('--store-shard-meta/--no-store-shard-meta', default=True, help='Store shard meta-data along with shard?')
@click.option('--lidar-dynamic-flag-bbox-padding-meters',
              type=float,
              help='Label BBOX padding distance (in meters) to enlarge bounding boxes for per-point dynamic-flag assignment',
              default=NVLabelProcessor.LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS)
@click.option('--global-speed-dynamic-threshold',
              type=float,
              help='Speed threshold (in meters/sec) to consider a moving object globally dynamic',
              default=NVLabelProcessor.GLOBAL_SPEED_DYNAMIC_THRESHOLD)
@click.option("--debug", is_flag=True, default=False, help="Enables debug logging outputs")
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """ Extracts a time-based subrange of data from NCore shards and outputs the data as a new shard """

    params = CLIBaseParams(**kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(level=logging.DEBUG if params.debug else logging.INFO,
                        format='<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s')

    ctx.obj = params


@cli.command()
@click.option('--start-timestamp-us',
              type=int,
              default=None,
              help="If provided, the start timestamp to restrict processing to")
@click.option('--end-timestamp-us',
              type=int,
              default=None,
              help="If provided, the end timestamp to restrict processing to")
@click.pass_context
def timestamps(ctx, *_, **kwargs) -> None:
    """Timestamp-based subrange selection"""

    pass


@cli.command()
@click.option('--seek-sec',
              type=click.FloatRange(min=0.0, max_open=True),
              help="Time to skip for the dataset conversion (in seconds)")
@click.option('--duration-sec',
              type=click.FloatRange(min=0.0, max_open=True),
              help="Restrict total duration of the dataset conversion (in seconds)")
@click.pass_context
def offset(ctx, seek_sec: float, duration_sec: float) -> None:
    """Offset-based subrange selection"""

    params: CLIBaseParams = ctx.obj

    # determine time-ranges from seek/duration relative to data
    loader = ShardDataLoader(ShardDataLoader.evaluate_shard_file_pattern(params.shard_file_pattern),
                             params.open_consolidated)

    start_timestamp_us, end_timestamp_us = time_bounds(loader.get_poses().T_rig_world_timestamps_us.tolist(), seek_sec,
                                                       duration_sec)

    if not (container_name := params.output_basename):
        container_name = '_'.join((str(x) for x in (loader.get_sequence_id(), start_timestamp_us, end_timestamp_us)))

    logging.debug(Path(params.output_dir) / (container_name + '.zarr.itar'))

    SubrangeDataWriter.process(
        loader,
        start_timestamp_us,
        end_timestamp_us,
        params.lidar_dynamic_flag_bbox_padding_meters,
        params.global_speed_dynamic_threshold,
        Path(params.output_dir),
        container_name,
        # TODO: add sensor selection
        None,
        None,
        None,
        params.store_shard_meta)


if __name__ == '__main__':
    cli(show_default=True)
