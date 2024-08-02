# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import click
import tqdm

import numpy as np

from ncore.impl.common.common import PoseInterpolator, time_bounds, HalfClosedInterval
from ncore.impl.data.data3 import ShardDataLoader, ContainerDataWriter
from ncore.impl.data.types import Poses, FrameTimepoint, FrameLabel3, Tracks, TrackLabel
from ncore.impl.common.nvidia_utils import LabelProcessor as NVLabelProcessor, load_maglev_egomotion
from ncore.impl.data_converter.data_converter import BaseNvidiaDataConverter
from ncore.impl.data_converter.waymo3 import WaymoConverter


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part"""

    shard_file_pattern: str
    skip_suffixes: Tuple[str]
    output_dir: str
    output_basename: Optional[str]
    open_consolidated: bool
    dynamic_flag_variant: str
    no_cameras: bool
    camera_ids: Tuple[str]
    no_lidars: bool
    lidar_ids: Tuple[str]
    no_radars: bool
    radar_ids: Tuple[str]
    egomotion_file: Optional[str]
    camera_frame_step: int
    lidar_frame_step: int
    radar_frame_step: int
    verbose: bool


@dataclass(kw_only=True, slots=True, frozen=True)
class DynamicFlagParameters:
    """Parameters used to compute dynamic flags"""

    label_ids_unconditionally_dynamic: set[str]
    label_ids_unconditionally_static: set[str]
    lidar_dynamic_flag_bbox_padding_meters: float
    global_speed_dynamic_threshold: float


class ChunkDataWriter:
    """Performs data subrange selection re-exports a new container with subselected data"""

    @staticmethod
    def process(
        # Source data + chunk range
        loader: ShardDataLoader,
        source_poses: Poses,
        source_egomotion_type: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
        camera_frame_step: int,
        lidar_frame_step: int,
        radar_frame_step: int,
        # Dynamic-flag processing parameters
        dynamic_flag_parameters: DynamicFlagParameters,
        # Output specification
        output_dir_path: Path,
        container_name: str,
        # Sensor selection (exports all sensors of a give type if not restricted)
        camera_ids: Optional[list[str]],
        lidar_ids: Optional[list[str]],
        radar_ids: Optional[list[str]],
    ) -> None:

        if camera_ids is None:
            camera_ids = loader.get_camera_ids()
        if lidar_ids is None:
            lidar_ids = loader.get_lidar_ids()
        if radar_ids is None:
            radar_ids = loader.get_radar_ids()

        assert start_timestamp_us < end_timestamp_us, "invalid time bounds"
        chunk_interval_us = HalfClosedInterval(
            start_timestamp_us, end_timestamp_us + 1
        )  # make sure to include end-timestamp in interval

        logging.debug(f"Writing chunk export to {output_dir_path / (container_name + '.zarr.itar')}")

        # ContainerDataWriter for all outputs (always single-shard)
        data_writer = ContainerDataWriter(
            output_dir_path,
            container_name,
            camera_ids,
            lidar_ids,
            radar_ids,
            loader.get_calibration_type(),
            source_egomotion_type,
            container_name,
            loader.get_generic_meta_data(),
            # always single-shard
            0,
            1,
            False,
        )

        ## Process poses

        # subselect poses
        target_poses_range = chunk_interval_us.cover_range(source_poses.T_rig_world_timestamps_us)

        assert (
            len(target_poses_range) > 1
        ), "insufficient pose samples - too restrictive time-range, require at least two poses"

        target_poses = Poses(
            source_poses.T_rig_world_base,
            source_poses.T_rig_worlds[target_poses_range],
            source_poses.T_rig_world_timestamps_us[target_poses_range],
        )

        # this is the slightly tighter interval (compared to chunk-interval) for which egomotion poses can be interpolated
        pose_interval_us = HalfClosedInterval(
            int(target_poses.T_rig_world_timestamps_us[0]), int(target_poses.T_rig_world_timestamps_us[-1] + 1)
        )  # make sure to include end-timestamp in interval

        # use pose interpolator over *target* pose range to make sure all frame-associated poses are within egomotion range
        pose_interpolator = PoseInterpolator(target_poses.T_rig_worlds, target_poses.T_rig_world_timestamps_us)

        data_writer.store_poses(target_poses)

        ## Process cameras
        for camera_id in tqdm.tqdm(camera_ids, desc="Cameras"):
            camera_sensor = loader.get_camera_sensor(camera_id)

            # subselect frames
            source_frame_timestamps_us = camera_sensor.get_frames_timestamps_us()
            target_frames_range = pose_interval_us.cover_range(source_frame_timestamps_us)

            # store subselected frames
            chunk_frame_index = 0
            chunk_frame_end_timestamps_us: list[int] = []
            for source_frame_idx in tqdm.tqdm(target_frames_range[::camera_frame_step], desc="Frames", leave=False):
                timestamps_us = np.stack(
                    (
                        camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                        camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                    )
                )

                # allow skipping frames (usually at start / end of range due to mixed sensor / pose frequencies) in case they are not within egomotion time-range
                if not pose_interpolator.in_range(timestamps_us):
                    logging.warning(
                        f"Skipping frame of camera {camera_id} as frame timestamps {timestamps_us} not "
                        f"in egomotion time-range [{pose_interpolator.start_timestamp}-{pose_interpolator.start_timestamp}]"
                    )
                    continue

                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                encoded_image_data = camera_sensor.get_frame_handle(source_frame_idx).get_data()

                data_writer.store_camera_frame(
                    camera_id,
                    chunk_frame_index,
                    encoded_image_data.get_encoded_image_data(),
                    encoded_image_data.get_encoded_image_format(),
                    T_rig_worlds,
                    timestamps_us,
                    {
                        name: camera_sensor.get_frame_generic_data(source_frame_idx, name)
                        for name in camera_sensor.get_frame_generic_data_names(source_frame_idx)
                    },
                    camera_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

                chunk_frame_index += 1
                chunk_frame_end_timestamps_us.append(timestamps_us[1])

            data_writer.store_camera_meta(
                camera_id,
                np.array(chunk_frame_end_timestamps_us, dtype=np.uint64),
                camera_sensor.get_T_sensor_rig(),
                camera_sensor.get_camera_model_parameters(),
                camera_sensor.get_camera_mask_image(),
                camera_sensor.get_generic_meta_data(),
            )

        ## Process radars
        for radar_id in tqdm.tqdm(radar_ids, desc="Radars"):
            radar_sensor = loader.get_radar_sensor(radar_id)

            # subselect frames
            source_frame_timestamps_us = radar_sensor.get_frames_timestamps_us()
            target_frames_range = pose_interval_us.cover_range(source_frame_timestamps_us)

            # store subselected frames
            chunk_frame_index = 0
            chunk_frame_end_timestamps_us = []
            for source_frame_idx in tqdm.tqdm(target_frames_range[::radar_frame_step], desc="Frames", leave=False):
                timestamps_us = np.stack(
                    (
                        radar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                        radar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                    )
                )

                # allow skipping frames (usually at start / end of range due to mixed sensor / pose frequencies) in case they are not within egomotion time-range
                if not pose_interpolator.in_range(timestamps_us):
                    logging.warning(
                        f"Skipping frame of radar {radar_id} as frame timestamps {timestamps_us} not "
                        f"in egomotion time-range [{pose_interpolator.start_timestamp}-{pose_interpolator.start_timestamp}]"
                    )
                    continue

                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                data_writer.store_radar_frame(
                    radar_id,
                    chunk_frame_index,
                    radar_sensor.get_frame_data(source_frame_idx, "xyz_s"),
                    radar_sensor.get_frame_data(source_frame_idx, "xyz_e"),
                    T_rig_worlds,
                    timestamps_us,
                    {
                        name: radar_sensor.get_frame_generic_data(source_frame_idx, name)
                        for name in radar_sensor.get_frame_generic_data_names(source_frame_idx)
                    },
                    radar_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

                chunk_frame_index += 1
                chunk_frame_end_timestamps_us.append(timestamps_us[1])

            data_writer.store_radar_meta(
                radar_id,
                np.array(chunk_frame_end_timestamps_us, dtype=np.uint64),
                radar_sensor.get_T_sensor_rig(),
                radar_sensor.get_generic_meta_data(),
            )

        ## Process lidars

        # Iterate once over all frames to collect surviving tracks / frame labels
        target_track_labels: dict[str, TrackLabel] = {}
        target_frame_labels: dict[str, dict[int, list[FrameLabel3]]] = {}
        for lidar_id in tqdm.tqdm(lidar_ids, desc="Lidar Labels"):
            lidar_sensor = loader.get_lidar_sensor(lidar_id)

            if lidar_id not in target_frame_labels:
                target_frame_labels[lidar_id] = {}

            # subselect frames
            source_frame_timestamps_us = lidar_sensor.get_frames_timestamps_us()
            target_frames_range = pose_interval_us.cover_range(source_frame_timestamps_us)

            # extract labels from subselected frames
            for source_frame_idx in tqdm.tqdm(
                target_frames_range[::lidar_frame_step], desc="Frame Labels", leave=False
            ):

                source_frame_timestamp_us = int(source_frame_timestamps_us[source_frame_idx])

                # allow skipping frames (usually at start / end of range due to mixed sensor / pose frequencies) in case they are not within egomotion time-range
                if not pose_interpolator.in_range(source_frame_timestamp_us):
                    logging.warning(
                        f"Skipping frame labels of lidar {lidar_id} as frame timestamps {source_frame_timestamp_us} not "
                        f"in egomotion time-range [{pose_interpolator.start_timestamp}-{pose_interpolator.start_timestamp}]"
                    )
                    continue

                if source_frame_timestamp_us not in target_frame_labels[lidar_id]:
                    target_frame_labels[lidar_id][source_frame_timestamp_us] = []

                for frame_label in lidar_sensor.get_frame_labels(source_frame_idx):
                    target_frame_labels[lidar_id][source_frame_timestamp_us].append(frame_label)

                    track_id = frame_label.track_id

                    if track_id not in target_track_labels:
                        target_track_labels[track_id] = TrackLabel(sensors={})

                    if lidar_id not in target_track_labels[track_id].sensors:
                        target_track_labels[track_id].sensors[lidar_id] = []

                    target_track_labels[track_id].sensors[lidar_id].append(source_frame_timestamp_us)

        target_track_dynamic_flag = NVLabelProcessor.track_global_dynamic_flag(
            target_frame_labels,
            label_strings_unconditionally_dynamic=dynamic_flag_parameters.label_ids_unconditionally_dynamic,
            label_strings_unconditionally_static=dynamic_flag_parameters.label_ids_unconditionally_static,
            global_speed_dynamic_threshold=dynamic_flag_parameters.global_speed_dynamic_threshold,
        )

        # Second iteration: store frames
        for lidar_id in tqdm.tqdm(lidar_ids, desc="Lidars"):
            lidar_sensor = loader.get_lidar_sensor(lidar_id)

            # subselect frames
            source_frame_timestamps_us = lidar_sensor.get_frames_timestamps_us()
            target_frames_range = pose_interval_us.cover_range(source_frame_timestamps_us)

            # store subselected frames
            chunk_frame_index = 0
            chunk_frame_end_timestamps_us = []
            for source_frame_idx in tqdm.tqdm(target_frames_range[::lidar_frame_step], desc="Frames", leave=False):
                timestamps_us = np.stack(
                    (
                        lidar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                        lidar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                    )
                )

                # allow skipping frames (usually at start / end of range due to mixed sensor / pose frequencies) in case they are not within egomotion time-range
                if not pose_interpolator.in_range(timestamps_us):
                    logging.warning(
                        f"Skipping frame of lidar {lidar_id} as frame timestamps {timestamps_us} not "
                        f"in egomotion time-range [{pose_interpolator.start_timestamp}-{pose_interpolator.start_timestamp}]"
                    )
                    continue

                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                # re-estimate dynamic flags based on local track data
                xyz_e = lidar_sensor.get_frame_data(source_frame_idx, "xyz_e")
                dynamic_flag, frame_labels = NVLabelProcessor.lidar_dynamic_flag(
                    lidar_id,
                    xyz_e,
                    timestamps_us[1],
                    target_frame_labels,
                    target_track_dynamic_flag,
                    lidar_dynamic_flag_bbox_padding_meters=dynamic_flag_parameters.lidar_dynamic_flag_bbox_padding_meters,
                )

                data_writer.store_lidar_frame(
                    lidar_id,
                    chunk_frame_index,
                    lidar_sensor.get_frame_data(source_frame_idx, "xyz_s"),
                    xyz_e,
                    lidar_sensor.get_frame_data(source_frame_idx, "intensity"),
                    lidar_sensor.get_frame_data(source_frame_idx, "timestamp_us"),
                    frame_labels,
                    T_rig_worlds,
                    timestamps_us,
                    {
                        name: lidar_sensor.get_frame_generic_data(source_frame_idx, name)
                        for name in lidar_sensor.get_frame_generic_data_names(source_frame_idx)
                    }
                    | {"dynamic_flag": dynamic_flag.astype(np.int8)},
                    lidar_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

                chunk_frame_index += 1
                chunk_frame_end_timestamps_us.append(timestamps_us[1])

        data_writer.store_lidar_meta(
            lidar_id,
            np.array(chunk_frame_end_timestamps_us, dtype=np.uint64),
            lidar_sensor.get_T_sensor_rig(),
            lidar_sensor.get_generic_meta_data(),
        )

        # store tracks
        data_writer.store_tracks(tracks=Tracks(track_labels=target_track_labels))

        ## Finalize output
        logging.info(f"Wrote chunk to {data_writer.finalize()}")


def get_dynamic_flag_parameters(variant: str, loader: ShardDataLoader) -> DynamicFlagParameters:
    """Provides the dynamic flag parameters for the chosen variant ['auto', 'nv', 'waymo']"""
    nv_params = DynamicFlagParameters(
        label_ids_unconditionally_dynamic=NVLabelProcessor.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC,
        label_ids_unconditionally_static=NVLabelProcessor.LABEL_STRINGS_UNCONDITIONALLY_STATIC,
        lidar_dynamic_flag_bbox_padding_meters=NVLabelProcessor.LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS,
        global_speed_dynamic_threshold=NVLabelProcessor.GLOBAL_SPEED_DYNAMIC_THRESHOLD,
    )

    waymo_params = DynamicFlagParameters(
        label_ids_unconditionally_dynamic=WaymoConverter.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC,
        label_ids_unconditionally_static=WaymoConverter.LABEL_STRINGS_UNCONDITIONALLY_STATIC,
        lidar_dynamic_flag_bbox_padding_meters=WaymoConverter.LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS,
        global_speed_dynamic_threshold=WaymoConverter.GLOBAL_SPEED_DYNAMIC_THRESHOLD,
    )

    match variant:
        case "nv":
            return nv_params
        case "waymo":
            return waymo_params
        case "auto":
            # try by matching calibration-type
            if input_calibration_type := loader.get_calibration_type() in ["scene-calib", "deepmap", "carter"]:
                logging.info("Auto-detected NV dynamic flag parameters")
                return nv_params

            if input_calibration_type in ["waymo-calibration"]:
                logging.info("Auto-detected Waymo dynamic flag parameters")
                return waymo_params

            # try by matching camera sensor-names
            if input_sensor_ids := set(loader.get_camera_ids()) & (
                set(BaseNvidiaDataConverter.Hyperion8Constants.CAMERAID_TO_RIGNAME.keys())
                | set(BaseNvidiaDataConverter.Hyperion81Constants.CAMERAID_TO_RIGNAME.keys())
            ):
                logging.info("Auto-detected NV dynamic flag parameters")
                return nv_params

            if input_sensor_ids & set(WaymoConverter.CAMERA_MAP.keys()):
                logging.info("Auto-detected Waymo dynamic flag parameters")
                return waymo_params

    raise RuntimeError(
        "Detecting dynamic flag parameters failed, consider extending lookup or specify supported variant explicitly via '--dynamic-flag-variant' parameter"
    )


@click.group()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option(
    "--skip-suffix",
    "skip_suffixes",
    multiple=True,
    type=str,
    help="Shard suffixes to skip",
    default=None,
)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--output-basename",
    type=str,
    default=None,
    help="Basename of the generated file - <sequence-id>@<start-time-us>-<end-time-us> will be used by default if not provided",
    required=False,
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option(
    "--dynamic-flag-variant",
    type=click.Choice(["auto", "nv", "waymo"], case_sensitive=False),
    default="auto",
    help="Variant-specific parameters to use for dynamic-flag assignment (auto exit with an error if variant lookup fails)",
)
@click.option("--no-cameras", is_flag=True, default=False, help="Disable exporting of any camera sensor")
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-lidars", is_flag=True, default=False, help="Disable exporting of any lidar sensor")
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-radars", is_flag=True, default=False, help="Disable exporting of any radar sensor")
@click.option(
    "--radar-id",
    "radar_ids",
    multiple=True,
    type=str,
    help="Radars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option(
    "--egomotion-file",
    type=str,
    help="If provided, overwrite egomotion poses with trajectory at file location (NV maglev pose format)",
    default=None,
)
@click.option(
    "--camera-frame-step",
    type=click.IntRange(min=1, max_open=True),
    help="Frame step for camera subsampling during dataset conversion",
)
@click.option(
    "--lidar-frame-step",
    type=click.IntRange(min=1, max_open=True),
    help="Frame step for lidar subsampling during dataset conversion",
)
@click.option(
    "--radar-frame-step",
    type=click.IntRange(min=1, max_open=True),
    help="Frame step for radar subsampling during dataset conversion",
)
@click.option("--verbose", is_flag=True, default=False, help="Enables debug logging outputs")
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Extracts a time-based subrange of data from NCore shards and outputs the data as a new shard"""

    params = CLIBaseParams(**kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(
        level=logging.DEBUG if params.verbose else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )

    ctx.obj = params


def ncore_chunk(
    params: CLIBaseParams,
    loader: ShardDataLoader,
    poses: Poses,
    egomotion_type: str,
    start_timestamp_us: int,
    end_timestamp_us: int,
) -> None:
    """Execute common components of chunk export"""
    # Output container name
    if not (container_name := params.output_basename):
        container_name = f"{loader.get_sequence_id()}@{str(start_timestamp_us)}-{str(end_timestamp_us)}"

    # Sensor selection
    camera_ids = list(params.camera_ids) if len(params.camera_ids) else None
    if params.no_cameras:
        camera_ids = []

    lidar_ids = list(params.lidar_ids) if len(params.lidar_ids) else None
    if params.no_lidars:
        lidar_ids = []

    radar_ids = list(params.radar_ids) if len(params.radar_ids) else None
    if params.no_radars:
        radar_ids = []

    ChunkDataWriter.process(
        loader,
        poses,
        egomotion_type,
        start_timestamp_us,
        end_timestamp_us,
        params.camera_frame_step,
        params.lidar_frame_step,
        params.radar_frame_step,
        get_dynamic_flag_parameters(params.dynamic_flag_variant, loader),
        Path(params.output_dir),
        container_name,
        camera_ids,
        lidar_ids,
        radar_ids,
    )


def get_poses(loader: ShardDataLoader, egomotion_file: Optional[str]) -> Tuple[Poses, str]:
    """Loads poses to be used as reference time-range (either from source data or egomotion-file overwrite)"""
    if not egomotion_file:
        # No overwrite
        return loader.get_poses(), loader.get_egomotion_type()

    # Load sensor extrinsics
    T_rig_sensors_base = {
        sensor_id: loader.get_sensor(sensor_id).get_T_rig_sensor() for sensor_id in loader.get_sensor_ids()
    }

    T_rig_worlds, T_rig_world_timestamps_us, egomotion_type = load_maglev_egomotion(
        T_rig_sensors_base, Path(egomotion_file)
    )

    assert len(T_rig_worlds), "No valid egomotion poses loaded"

    # Stack all poses (common canonical format convention)
    T_rig_worlds = np.stack(T_rig_worlds)
    T_rig_world_timestamps_us = np.array(T_rig_world_timestamps_us, dtype=np.uint64)

    # Select first pose as base pose
    T_rig_world_base = T_rig_worlds[0]
    T_rig_worlds = np.linalg.inv(T_rig_world_base) @ T_rig_worlds

    # Assemble Poses struct
    return (
        Poses(
            T_rig_world_base=T_rig_world_base,
            T_rig_worlds=T_rig_worlds,
            T_rig_world_timestamps_us=T_rig_world_timestamps_us,
        ),
        egomotion_type,
    )


@cli.command()
@click.option("--start-timestamp-us", type=int, help="If provided, the start timestamp to restrict processing to")
@click.option("--end-timestamp-us", type=int, help="If provided, the end timestamp to restrict processing to")
@click.pass_context
def timestamps(ctx, start_timestamp_us: int | None, end_timestamp_us: int | None) -> None:
    """Timestamp-based subrange selection"""

    params: CLIBaseParams = ctx.obj

    # determine time-ranges from seek/duration relative to data
    loader = ShardDataLoader(
        ShardDataLoader.evaluate_shard_file_pattern(params.shard_file_pattern, params.skip_suffixes),
        params.open_consolidated,
    )

    poses, egomotion_type = get_poses(loader, params.egomotion_file)

    ncore_chunk(
        params,
        loader,
        poses,
        egomotion_type,
        start_timestamp_us if start_timestamp_us else int(poses.T_rig_world_timestamps_us[0]),
        end_timestamp_us if end_timestamp_us else int(poses.T_rig_world_timestamps_us[-1]),
    )


@cli.command()
@click.option(
    "--seek-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Time to skip for the dataset conversion (in seconds)",
)
@click.option(
    "--duration-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Restrict total duration of the dataset conversion (in seconds)",
)
@click.pass_context
def offset(ctx, seek_sec: float | None, duration_sec: float | None) -> None:
    """Offset-based subrange selection"""

    params: CLIBaseParams = ctx.obj

    # determine time-ranges from seek/duration relative to data
    loader = ShardDataLoader(
        ShardDataLoader.evaluate_shard_file_pattern(params.shard_file_pattern, params.skip_suffixes),
        params.open_consolidated,
    )

    poses, egomotion_type = get_poses(loader, params.egomotion_file)

    start_timestamp_us, end_timestamp_us = time_bounds(poses.T_rig_world_timestamps_us.tolist(), seek_sec, duration_sec)

    ncore_chunk(
        params,
        loader,
        poses,
        egomotion_type,
        start_timestamp_us,
        end_timestamp_us,
    )


if __name__ == "__main__":
    cli(show_default=True)
