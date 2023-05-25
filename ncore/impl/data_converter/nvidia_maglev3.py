# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import json
import tempfile

from pathlib import Path
from typing import Optional

import numpy as np
import tqdm
import point_cloud_utils as pcu

from ncore.impl.data_converter.data_converter import BaseNvidiaDataConverter
from ncore.impl.data.data3 import ContainerDataWriter
from ncore.impl.data.types import FThetaCameraModelParameters, LabelSource, Poses, ShutterType

from ncore.impl.common.nvidia_utils import (load_maglev_camera_indexer_frame_meta, load_maglev_lidar_indexer_frame_meta,
                                           load_maglev_egomotion, load_maglev_session_id, parse_rig_sensors_from_dict,
                                           sensor_to_rig, LabelProcessor, camera_intrinsic_parameters,
                                           compute_fw_polynomial, compute_ftheta_parameters, camera_car_mask,
                                           vehicle_bbox)
from ncore.impl.common.common import load_jsonl, PoseInterpolator, uniform_subdivide_range, SimpleTimer
from ncore.impl.av_utils import isWithin3DBBox


class NvidiaMaglevConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data conversion (based on Maglev ncore-pp workflows data extraction)
    """

    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.seek_sec: float = config.seek_sec
        self.duration_sec: float = config.duration_sec

        self.shard_id: int = config.shard_id
        self.shard_count: int = config.shard_count

        self.egomotion_file : Optional[Path] = Path(config.egomotion_file) if config.egomotion_file else None

        self.skip_dynamic_flag: bool = config.skip_dynamic_flag

    @staticmethod
    def get_sequence_paths(config) -> list[Path]:
        return [Path(config.root_dir)]

    @staticmethod
    def from_config(config) -> NvidiaMaglevConverter:
        return NvidiaMaglevConverter(config)

    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs the conversion of a single session (single job output of Maglev ncore-pp workflow)
        """

        self.sequence_path = sequence_path

        # Read rig json file and sensor information
        with open(sequence_path / 'rig.json', 'r') as fp:
            self.rig = json.load(fp)

        self.sensors_calibration_data = parse_rig_sensors_from_dict(self.rig)

        # Determine session-id to be processed
        session_id = load_maglev_session_id(self.sequence_path)
        self.logger.info(f'Converting session {session_id} [shard {self.shard_id}/{self.shard_count}]')

        # DataWriter for all outputs
        self.data_writer = ContainerDataWriter(
            self.output_dir / session_id,
            f'{session_id}_{self.shard_id}-{self.shard_count}',
            list(self.CAMERAID_TO_RIGNAME.keys()),
            list(self.LIDARID_TO_RIGNAME.keys()),
            []  # no radars yet
            ,
            # TODO: parse these from the data
            'scene-calib',
            'lidar-egomotion',
            session_id,
            self.shard_id,
            self.shard_count,
            True)

        # Decode data from maglev
        self.decode_poses()

        self.decode_labels()

        self.decode_cameras()

        self.decode_lidars()

        # Store per-shard meta data / final success state / close file
        self.data_writer.finalize()

    def decode_poses(self):
        logger = self.logger.getChild('decode_poses')
        logger.info(f'Loading poses')

        # Load timestamped poses variables
        self.global_T_rig_worlds, self.global_T_rig_world_timestamps_us = load_maglev_egomotion(
            self.sequence_path, self.sensors_calibration_data, self.egomotion_file)

        assert len(self.global_T_rig_worlds), "No valid egomotion poses loaded"

        # Stack all poses (common canonical format convention)
        self.global_T_rig_worlds = np.stack(self.global_T_rig_worlds)
        self.global_T_rig_world_timestamps_us = np.array(self.global_T_rig_world_timestamps_us, dtype=np.uint64)

        # Select reference base pose and convert all poses relative to this reference.
        # The base pose represents a worldToGlobal transformation and the first pose
        # of the trajectory defines the global frame of reference
        # (all other world poses are encoded relative to this global frame from here one,
        # allowing to represent, e.g., point world-coordinates in single f32 precision)
        T_rig_world_base = self.global_T_rig_worlds[0]
        self.global_T_rig_worlds = np.linalg.inv(T_rig_world_base) @ self.global_T_rig_worlds

        # Apply and remember global time-range restrictions for dataset (used for all pose interpolation within shard)
        global_target_start_timestamp_us, global_target_end_timestamp_us = self.time_bounds(self.global_T_rig_world_timestamps_us, self.seek_sec, self.duration_sec)
        global_range_start             = np.argmax(self.global_T_rig_world_timestamps_us >= global_target_start_timestamp_us)
        global_range_end               = np.argmin(self.global_T_rig_world_timestamps_us < global_target_end_timestamp_us) \
                                         if global_target_end_timestamp_us < self.global_T_rig_world_timestamps_us[-1] \
                                         else len(self.global_T_rig_world_timestamps_us) # full range of poses or restriction
        self.global_T_rig_worlds              = self.global_T_rig_worlds[global_range_start:global_range_end]
        self.global_T_rig_world_timestamps_us = self.global_T_rig_world_timestamps_us[global_range_start:global_range_end]
        self.global_start_timestamp_us        = self.global_T_rig_world_timestamps_us[0]
        self.global_end_timestamp_us          = self.global_T_rig_world_timestamps_us[-1]

        assert self.global_start_timestamp_us >= global_target_start_timestamp_us
        assert self.global_end_timestamp_us <= global_target_end_timestamp_us # note: global bounds are inclusive

        # Apply uniform subdivision for current shard to get local pose range with non-inclusive *single* pose **overlap**.
        # This guarantees that all frames can be associated with a unique shard (needs to be un-done when loading multi-shard sequences)
        #
        # Example
        # shard0_pose_timestamps = [0,1,2,3], valid pose-range-timestamps to select frame data [0,3) -> 0 <= t < 3
        # shard1_pose_timestamps = [3,4,5,6], valid pose-range-timestamps to select frame data [3,6) -> 3 <= t < 6
        local_range, _ = uniform_subdivide_range(self.shard_id, self.shard_count,
                                                 # *skip* first global pose unconditionally from all local-ranges to be sure we can interpolate within initial frames,
                                                 # so first shard's pose range in the example above will really be [1,3) -> 1 <= t < 3
                                                 1,
                                                 len(self.global_T_rig_world_timestamps_us))
        # extend local range by single non-inclusive pose to keep in local shard
        local_range_start = local_range[0]
        local_range_end = min(
            # non-extended local range end
            (local_range[-1] + 1)
            # extend by single additional non-inclusive pose
            + 1,
            len(self.global_T_rig_world_timestamps_us))
        local_T_rig_worlds = self.global_T_rig_worlds[local_range_start:local_range_end]
        local_T_rig_world_timestamps_us = self.global_T_rig_world_timestamps_us[local_range_start:local_range_end]
        self.local_start_timestamp_us = local_T_rig_world_timestamps_us[0]
        self.local_end_timestamp_us = local_T_rig_world_timestamps_us[-1]

        logger.debug(
            f'shard {self.shard_id+1}/{self.shard_count} | local_range_start {local_range_start} / local_range_end {local_range_end} | '
            f'{self.local_start_timestamp_us} <= t < {self.local_end_timestamp_us}')

        assert self.local_start_timestamp_us >= self.global_start_timestamp_us
        assert self.local_end_timestamp_us <= self.global_end_timestamp_us  # note: global bounds are inclusive

        # Save the poses
        self.data_writer.store_poses(Poses(T_rig_world_base, local_T_rig_worlds, local_T_rig_world_timestamps_us))

        # Log base pose to share it more easily with downstream teams (it's serialized also explicitly)
        with np.printoptions(floatmode='unique', linewidth=200):  # print in highest precision
            logger.info(
                f'> processed {local_range_end - local_range_start} / {global_range_end - global_range_start} local / global poses, using base pose:\n{T_rig_world_base}')

    def decode_labels(self):
        logger = self.logger.getChild('decode_labels')
        logger.info(f'Loading labels')

        # Initialize annotation structs (defaults in case no labels are available loaded)
        self.track_labels: dict[int, dict] = {}
        self.frame_labels: dict[str, dict] = {}

        # Process autolabels, if available
        labels_path = self.sequence_path / 'cuboids_tracked' / 'labels_lidar.parquet'
        if not labels_path.exists():
            logger.warn(f'> file {labels_path} doesn\'t exist, skipping label generation')
            return

        # Perform label parsing (of global time)
        self.track_labels, self.frame_labels = LabelProcessor.parse(
            labels_path,
            {
                lidar_id: load_maglev_lidar_indexer_frame_meta(Path(self.sequence_path / 'lidars' / lidar_rig_name))
                for lidar_id, lidar_rig_name in self.LIDARID_TO_RIGNAME.items()
            },
            {
                lidar_id: sensor_to_rig(self.sensors_calibration_data[lidar_rig_name])
                for lidar_id, lidar_rig_name in self.LIDARID_TO_RIGNAME.items()
            },
            self.global_T_rig_world_timestamps_us,
            self.global_T_rig_worlds,
            LabelSource.AUTOLABEL,
            logger)

        # Save the accumulated tracks in global time
        self.data_writer.store_labels(self.track_labels)


    def decode_cameras(self):
        logger = self.logger.getChild('decode_cameras')
        logger.info(f'Loading camera data [shard {self.shard_id}/{self.shard_count}]')

        # Pose interpolator to obtain start / end egomotion poses
        pose_interpolator = PoseInterpolator(self.global_T_rig_worlds, self.global_T_rig_world_timestamps_us)

        # Process all camera sensors
        for camera_id, camera_rig_name in self.CAMERAID_TO_RIGNAME.items():
            logger.info(f'Processing camera {camera_rig_name}')

            camera_type = self.CAMERAID_TO_CAMERATYPE[camera_id]

            # Load frame numbers and timestamps
            raw_frame_numbers, raw_frame_timestamps_us = load_maglev_camera_indexer_frame_meta(
                self.sequence_path / 'cameras' / camera_rig_name)

            assert len(raw_frame_numbers) == len(raw_frame_timestamps_us)

            # Map raw frame timestamps to end-of-frame timestamps respecting exposure times in middle of row
            raw_frame_timestamps_us = raw_frame_timestamps_us - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type]

            # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
            global_range_start = np.argmax(raw_frame_timestamps_us >= self.global_start_timestamp_us)
            global_range_end = np.argmin(raw_frame_timestamps_us < self.global_end_timestamp_us) \
                if raw_frame_timestamps_us[-1] > self.global_end_timestamp_us else len(raw_frame_timestamps_us) # take all frames if all are within egomotion range, or determine last valid frame

            global_frame_timestamps_us = raw_frame_timestamps_us[global_range_start:global_range_end]

            # Subsample frames to valid local ranges
            local_range_start = np.argmax(raw_frame_timestamps_us >= self.local_start_timestamp_us)
            local_range_end = np.argmin(raw_frame_timestamps_us < self.local_end_timestamp_us) \
                if raw_frame_timestamps_us[-1] > self.local_end_timestamp_us else len(raw_frame_timestamps_us)

            local_frame_numbers = raw_frame_numbers[local_range_start:local_range_end]
            local_frame_timestamps_us = raw_frame_timestamps_us[local_range_start:local_range_end]

            logger.debug(
                f'camera {camera_rig_name} | local_range_start {local_range_start} / local_range_end {local_range_end} | '
                f'{self.local_start_timestamp_us} <= t < {self.local_end_timestamp_us}')

            assert global_frame_timestamps_us[0] <= local_frame_timestamps_us[0]
            assert local_frame_timestamps_us[1] <= global_frame_timestamps_us[-1]  # note: global bounds are inclusive

            assert self.local_start_timestamp_us <= local_frame_timestamps_us[0]
            if self.shard_id <  self.shard_count - 1:
                assert local_frame_timestamps_us[-1] < self.local_end_timestamp_us  # note: local bounds are non-inclusive in this regular case
            else:
                # last-shard or single-shard case
                assert local_frame_timestamps_us[-1] <= self.local_end_timestamp_us  # note: local bounds are inclusive in this end-case

            ## Compute sensor-specific data

            # Extract the calibration metadata
            camera_calibration_data = self.sensors_calibration_data[camera_rig_name]
            T_sensor_rig = sensor_to_rig(camera_calibration_data)

            # Estimate the forward polynomial
            intrinsic = camera_intrinsic_parameters(
                camera_calibration_data, logger
            )  # TODO: make sure we return 6th-order polynomial unconditionally. Ideally also cleanup clumpsy single-array representation for intrinsics

            bw_poly = intrinsic[4:]
            fw_poly = compute_fw_polynomial(intrinsic)
            _, max_angle = compute_ftheta_parameters(np.concatenate((intrinsic, fw_poly)), np.deg2rad(self.MAX_CAMERA_FOV_DEG / 2))

            # Constant mask image, which currently only contains the ego car mask
            # TODO: extend this with dynamic object masks
            mask_image = camera_car_mask(camera_calibration_data)

            # Local camera pose timestamps corresponding to end-of-frame timestamps
            local_camera_timestamps_us = np.stack(local_frame_timestamps_us) \
                if local_frame_timestamps_us.size != 0 else np.empty_like(local_frame_timestamps_us) # check that at least a single frame was processed

            self.data_writer.store_camera_meta(
                camera_id, local_camera_timestamps_us, T_sensor_rig,
                FThetaCameraModelParameters(intrinsic[2:4].astype(np.uint64), ShutterType.ROLLING_TOP_TO_BOTTOM,
                                            intrinsic[0:2],
                                            FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE, bw_poly,
                                            fw_poly, max_angle), mask_image.get_image())

            # Load tar file containing images
            tar_file = open(self.sequence_path / 'cameras' / camera_rig_name / 'images.tar', 'rb')
            tar_index = json.load(open(self.sequence_path / 'cameras' / camera_rig_name / 'images.tar.idx.json', 'r'))

            ## Process all valid images
            for continous_local_frame_index, (frame_number, frame_end_timestamp_us) in \
                tqdm.tqdm(enumerate(zip(local_frame_numbers, local_frame_timestamps_us)), total=len(local_frame_numbers)):

                # Load image file data from archive
                file_record = tar_index[f'./{str(frame_number)}.jpeg']
                tar_file.seek(file_record['offset_data'])
                image_file_binary_data = tar_file.read(file_record['size'])

                # Interpolate the start and end pose to the timestamps of the first and last row
                timestamps_us = np.array([
                    # Snap very first frame to the start of global egomotion in case timestamp of first row is outside of global pose-range
                    max(frame_end_timestamp_us - self.CAMERATYPE_TO_ROLLINGSHUTTERDELAY_US[camera_type],
                        self.global_T_rig_world_timestamps_us[0]),
                    frame_end_timestamp_us
                ])
                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                self.data_writer.store_camera_frame(camera_id, continous_local_frame_index, image_file_binary_data, 'jpeg', T_rig_worlds, timestamps_us)

            logger.info(f'> processed {len(local_frame_timestamps_us)} local'
                        f' images [shard {self.shard_id}/{self.shard_count}]')

        logger.info(f'> processed {len(self.CAMERAID_TO_RIGNAME)} cameras')


    def decode_lidars(self):
        logger = self.logger.getChild('decode_lidars')
        logger.info(f'Loading lidar data [shard {self.shard_id}/{self.shard_count}]')

        # Pose interpolator to obtain start / end egomotion poses
        pose_interpolator = PoseInterpolator(self.global_T_rig_worlds, self.global_T_rig_world_timestamps_us)

        # Load vehicle bounding box (defined in rig frame)
        vehicle_bbox_rig = vehicle_bbox(self.rig)
        vehicle_bbox_rig[3:6] += self.LIDAR_FILTER_VEHICLE_BBOX_PADDING_METERS  # pad the bounding box slightly

        # Process all lidar sensors
        for lidar_id, lidar_rig_name in self.LIDARID_TO_RIGNAME.items():
            logger.info(f'Processing lidar {lidar_rig_name}')

            # Load extrinsics
            T_sensor_rig = sensor_to_rig(self.sensors_calibration_data[lidar_rig_name])

            # Load frame numbers and timestamps
            frames_metadata = load_maglev_lidar_indexer_frame_meta(self.sequence_path / 'lidars' / lidar_rig_name)
            raw_frame_numbers = frames_metadata.frame_numbers
            raw_frame_timestamps_us = frames_metadata.frame_endtimes_us
            raw_frame_egocompensated = np.full_like(raw_frame_timestamps_us,
                                                    frames_metadata.frames_egocompensated,
                                                    dtype=np.bool8)
            del (frames_metadata)

            assert len(raw_frame_numbers) == len(raw_frame_timestamps_us)

            # Get the frame range of the first and last frame relative to available egomotion poses
            global_range_start = np.argmax(raw_frame_timestamps_us >= self.global_start_timestamp_us)
            global_range_end = np.argmin(raw_frame_timestamps_us < self.global_end_timestamp_us) \
                if raw_frame_timestamps_us[-1] > self.global_end_timestamp_us else len(raw_frame_timestamps_us)

            global_frame_numbers = raw_frame_numbers[global_range_start:global_range_end]
            global_frame_timestamps_us = raw_frame_timestamps_us[global_range_start:global_range_end]

            # Subsample frames to valid local ranges
            local_range_start = np.argmax(raw_frame_timestamps_us >= self.local_start_timestamp_us)
            local_range_end = np.argmin(raw_frame_timestamps_us < self.local_end_timestamp_us) \
                if raw_frame_timestamps_us[-1] > self.local_end_timestamp_us else len(raw_frame_timestamps_us)

            local_frame_numbers = raw_frame_numbers[local_range_start:local_range_end]
            local_frame_timestamps_us = raw_frame_timestamps_us[local_range_start:local_range_end] # corresponds to end-of-frame
            local_frame_egocompensated = raw_frame_egocompensated[local_range_start:local_range_end]
            num_local_frames = len(local_frame_numbers)

            logger.debug(
                f'lidar {lidar_rig_name} | local_range_start {local_range_start} / local_range_end {local_range_end} | '
                f'{self.local_start_timestamp_us} <= t < {self.local_end_timestamp_us}')

            assert global_frame_timestamps_us[0] <= local_frame_timestamps_us[0]
            assert local_frame_timestamps_us[1] <= global_frame_timestamps_us[-1]  # note: global bounds are inclusive

            assert self.local_start_timestamp_us <= local_frame_timestamps_us[0]
            if self.shard_id <  self.shard_count - 1:
                assert local_frame_timestamps_us[-1] < self.local_end_timestamp_us  # note: local bounds are non-inclusive in this regular case
            else:
                # last-shard or single-shard case
                assert local_frame_timestamps_us[-1] <= self.local_end_timestamp_us  # note: local bounds are inclusive in this end-case

            # Store all static sensor data
            self.data_writer.store_lidar_meta(lidar_id, local_frame_timestamps_us, T_sensor_rig)

            # Load tar file containing frames
            tar_file = open(self.sequence_path / 'lidars' / lidar_rig_name / 'frames.tar', 'rb')
            tar_index = json.load(open(self.sequence_path / 'lidars' / lidar_rig_name / 'frames.tar.idx.json', 'r'))

            ## Process all valid point clouds
            for continuous_local_frame_index, (frame_number, frame_end_timestamp_us, frame_egocompensated) in \
                tqdm.tqdm(enumerate(zip(local_frame_numbers, local_frame_timestamps_us, local_frame_egocompensated)), total=num_local_frames):

                # Interpolate egomotion at frame end timestamp for sensor reference pose at end-of-spin time
                T_world_sensorRef = np.linalg.inv(pose_interpolator.interpolate_to_timestamps(frame_end_timestamp_us)[0] @ T_sensor_rig)

                # Start timer
                timer = SimpleTimer()

                # Load point clouds / ply files from archive
                file_record = tar_index[f'./{str(frame_number)}.ply']
                tar_file.seek(file_record['offset_data'])
                ply_binary_data = tar_file.read(file_record['size'])

                # Need a temporary file-system file as PCU can't load from memory
                with tempfile.NamedTemporaryFile(suffix='.ply') as ply:
                    ply.write(ply_binary_data)
                    mesh = pcu.load_triangle_mesh(ply.name)

                time_load = timer.elapsed_sec(restart = True)

                # Remove all points with *duplicate* coordinates (these seem to be present in the input already)
                # and remember the indices of the valid points to load other attributes
                xyz, unique_input_idxs = np.unique(mesh.vertex_data.positions, axis=0, return_index=True) # Motion-compensated end-points in end-of-spin frame
                intensity = mesh.vertex_data.custom_attributes['intensity'][unique_input_idxs].flatten() / 2.55 # Intensities are oddly represented as [0.0 .. 2.55], normalize to [0.0 .. 1.0]
                point_count = xyz.shape[0]

                # Create 3D ray structure of 3D rays in sensor space with accompanying metadata.
                # Columns; xyz_s, xyz_e, intensity, dynamic_flag, timestamp
                # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic

                ## Determine ray start-point by interpolating poses at per-point timestamps

                # Switch on different storage variants
                ts_lo = ts_hi = None
                if all(key in mesh.vertex_data.custom_attributes for key in ('timestamp_lo', 'timestamp_hi')):
                    # CSFT-based timestamp storage
                    ts_lo = np.array(mesh.vertex_data.custom_attributes['timestamp_lo'])[unique_input_idxs].astype(np.uint32)
                    ts_hi = np.array(mesh.vertex_data.custom_attributes['timestamp_hi'])[unique_input_idxs].astype(np.uint32)
                if auxChannelsData := mesh.aux_data.get('auxChannelsData', None):
                    if all(key in auxChannelsData for key in ('time_lo', 'time_hi')):
                        # Lidar-exporter-based timestamp storage
                        ts_lo = np.array(auxChannelsData['time_lo'])[unique_input_idxs].astype(np.uint32)
                        ts_hi = np.array(auxChannelsData['time_hi'])[unique_input_idxs].astype(np.uint32)

                if ts_lo is not None and ts_hi is not None:
                    # Perform time-dependent per sample start-point interpolation,
                    # stitching together uin64 timestamps from lo/hi parts
                    timestamp = (np.left_shift(ts_hi, 32, dtype=np.uint64) + ts_lo).flatten()
                    del(ts_lo, ts_hi)

                    # Special case: allow snapping to end-of-frame timestamp for *initial* frames as
                    # valid egomotion (in particular lidar-based egomotion) might not have been
                    # evaluated in the past before the processed lidar frame's end-of-frame timestamp
                    # (usually at start of sequence)
                    if frame_number == global_frame_numbers[0]:
                        past_idxs = timestamp < self.global_T_rig_world_timestamps_us[0]

                        if np.any(past_idxs):
                            logger.info("> snapping out-of-range point timestamps of *initial* spin to start of egomotion")
                            timestamp[past_idxs] = frame_end_timestamp_us

                    # Special case: snap too far in the future point timestamps to last valid end-of-frame timestamp
                    elif frame_number == global_frame_numbers[-1]:
                        future_idxs = timestamp > self.global_T_rig_world_timestamps_us[-1]

                        if np.any(future_idxs):
                            logger.info("> snapping out-of-range point timestamps of *last* spins to end of egomotion")
                            timestamp[future_idxs] = frame_end_timestamp_us

                    # Determine unique timestamps to only perform actually required pose interpolations (a lot of points share the same timestamp)
                    timestamp_unique, unique_timestamp_reverse_idxs = np.unique(timestamp, return_inverse=True)

                    # Lidar frame poses for each point (will throw in case invalid timestamps are loaded) expressed in the reference sensor's frame
                    # sensor_sensorRef = world_sensorRef * sensor_world = world_sensorRef * (rig_world * sensor_rig)
                    T_sensor_sensorRef_unique = T_world_sensorRef @ pose_interpolator.interpolate_to_timestamps(timestamp_unique) @ T_sensor_rig

                    # Pick sensor positions (in end-of-spin reference pose) for each start point (blow up to original potentially non-unique timestamp range)
                    xyz_s = T_sensor_sensorRef_unique[unique_timestamp_reverse_idxs, :3, -1]  # N x 3

                    if not frame_egocompensated:
                        # Need to motion-compensate ray end points into reference frame (they are defined in the time-dependent sensor-frames).
                        # "Apply" the full homogeneous T_sensor_sensorRef (blowing up to non-unique version for each point) transformation by
                        # performing rotation of source points and reusing the already extracted translation components of ray start points
                        xyz = (T_sensor_sensorRef_unique[unique_timestamp_reverse_idxs, :3, :3]
                               @ xyz[:, :, None]).squeeze(-1) + xyz_s  # N x 3

                    del(timestamp_unique, unique_timestamp_reverse_idxs, T_sensor_sensorRef_unique)

                else:
                    if continuous_local_frame_index == 0:  # Warn once in first iteration only
                        logger.warn('> no lidar point timestamps available (missing \'timestamp_lo\' / '
                                    ' \'timestamp_hi\' attributes), falling back to *constant* lidar start points')

                    if not frame_egocompensated:
                        raise ValueError('egomotion-compensated input point-cloud required if no timestamps are available')

                    # No per-point timestamps available, fallback to using *constant*
                    # lidar origin as start point for all rays
                    timestamp = np.full((point_count), frame_end_timestamp_us, dtype=np.uint64)
                    xyz_s = np.full((point_count, 3), [0, 0, 0], dtype=np.float32)  # N x 3

                del(mesh)

                # Homogeneous ray end points in sensor frame
                xyz_e = np.row_stack([xyz.transpose(), np.ones(point_count, dtype=np.float32)])  # 4 x N

                # Transform points from sensor to rig frame
                xyz_e_rig = T_sensor_rig @ xyz_e

                # Filter points inside the vehicles bounding-box
                valid_idxs = np.logical_not(isWithin3DBBox(xyz_e_rig[0:3, :].transpose(), vehicle_bbox_rig.reshape(1,-1)))

                # Drop homogenous dimension and transpose to match output dimension
                xyz_e = xyz_e[:-1, :].transpose()  # N x 3

                # Compute distances and filter points based on max distance
                valid_idxs &= np.linalg.norm(xyz_s - xyz_e, axis=1) < self.LIDARID_TO_FILTER_MAX_DISTANCE_METERS[lidar_id]

                time_process = timer.elapsed_sec(restart = True)

                # Compute dynamic flag / load current frame labels
                dynamic_flag, frame_labels = LabelProcessor.lidar_dynamic_flag(lidar_id,
                                                                               xyz,
                                                                               frame_end_timestamp_us,
                                                                               self.track_labels,
                                                                               self.frame_labels,
                                                                               skip_dynamic_flag=self.skip_dynamic_flag)

                time_dynflag = timer.elapsed_sec(restart = True)

                # Subselect to valid points
                xyz_s = xyz_s[valid_idxs, :]
                xyz_e = xyz_e[valid_idxs, :]
                intensity = intensity[valid_idxs]
                dynamic_flag = dynamic_flag[valid_idxs]
                timestamp = timestamp[valid_idxs]

                # Interpolate start / end pose
                timestamps_us = np.array([np.min(timestamp), frame_end_timestamp_us])
                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                time_process += timer.elapsed_sec(restart = True)

                # Serialize lidar frame
                self.data_writer.store_lidar_frame(lidar_id,
                                                   continuous_local_frame_index,
                                                   xyz_s,
                                                   xyz_e,
                                                   intensity,
                                                   timestamp,
                                                   dynamic_flag,
                                                   None,
                                                   frame_labels,
                                                   T_rig_worlds,
                                                   timestamps_us)

                time_store = timer.elapsed_sec(restart = True)

                logger.debug(f'> spin {continuous_local_frame_index+1}/{num_local_frames} | load/process/dynflag/store {time_load:.2f}/{time_process:.2f}/{time_dynflag:.2f}/{time_store:.2f}sec')

            logger.info(f'> processed {len(local_frame_timestamps_us)} local'
                        f' point clouds [shard {self.shard_id}/{self.shard_count}]')

        logger.info(f'> processed {len(self.LIDARID_TO_RIGNAME)} lidars')
