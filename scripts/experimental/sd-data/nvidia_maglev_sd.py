# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import json

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tqdm
import dataclasses_json
import PIL.Image as PILImage
import webdataset as wds
import scipy

from src.dsai_internal.data_converter.data_converter import BaseNvidiaDataConverter
from src.dsai_internal.data import types, util

from src.dsai_internal.common.nvidia_utils import (load_maglev_camera_indexer_frame_meta, load_maglev_egomotion,
                                                   load_maglev_session_id, parse_rig_sensors_from_dict, sensor_to_rig,
                                                   camera_intrinsic_parameters, compute_fw_polynomial,
                                                   compute_ftheta_parameters)
from src.dsai_internal.common.common import uniform_subdivide_range


class PositionInterpolator:
    def __init__(self, poses, timestamps_us):

        # extract positions
        positions = poses[:, 0:3, 3]

        # create splines
        self.position_spline_x = scipy.interpolate.UnivariateSpline(
            timestamps_us,
            positions[:, 0],
            k=1,  # linear spline
            ext='raise',  # disallow extrapolation
            check_finite=True,
        )
        self.position_spline_y = scipy.interpolate.UnivariateSpline(
            timestamps_us,
            positions[:, 1],
            k=1,  # linear spline
            ext='raise',  # disallow extrapolation
            check_finite=True,
        )
        self.position_spline_z = scipy.interpolate.UnivariateSpline(
            timestamps_us,
            positions[:, 2],
            k=1,  # linear spline
            ext='raise',  # disallow extrapolation
            check_finite=True,
        )
        self.velocity_spline_x = self.position_spline_x.derivative()
        self.velocity_spline_y = self.position_spline_y.derivative()
        self.velocity_spline_z = self.position_spline_z.derivative()

    def get_positions(self, timestamps_us) -> np.ndarray:
        return np.column_stack([
            self.position_spline_x(timestamps_us),
            self.position_spline_y(timestamps_us),
            self.position_spline_z(timestamps_us)
        ])

    def get_velocities_m_us(self, timestamps_us) -> np.ndarray:
        return np.column_stack([
            self.velocity_spline_x(timestamps_us),
            self.velocity_spline_y(timestamps_us),
            self.velocity_spline_z(timestamps_us)
        ])

    def get_velocities_m_s(self, timestamps_us) -> np.ndarray:
        return self.get_velocities_m_us(timestamps_us) * 1e6  # convert from m/us to m/s

    def get_velocities_km_h(self, timestamps_us) -> np.ndarray:
        return self.get_velocities_m_s(timestamps_us) * 3.6  # convert from m/s to km/h

    def get_speeds_km_h(self, timestamps_us) -> np.ndarray:
        return np.linalg.norm(self.get_velocities_km_h(timestamps_us), axis=1)


class NvidiaMaglevConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data conversion (Stable-Diffusion data, based on Maglev data extraction)
    """
    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.seek_sec: float = config.seek_sec
        self.duration_sec: float = config.duration_sec

        self.shard_id: int = config.shard_id
        self.shard_count: int = config.shard_count

        self.wds_shard_maxsize_mib: int = config.wds_shard_maxsize_mib
        self.wds_shard_maxsamples: int = config.wds_shard_maxsamples

        self.min_speed_km_h: Optional[float] = config.min_speed_km_h

    @staticmethod
    def get_sequence_dirs(config) -> list[Path]:
        return [Path(config.root_dir)]

    @staticmethod
    def from_config(config) -> NvidiaMaglevConverter:
        return NvidiaMaglevConverter(config)

    def store_shard_meta(self, successful: bool) -> None:
        ''' Store shard-specific meta-data '''
        with open(self.output_dir / self.session_id / f'shard-meta-{util.padded_index_string(self.shard_id, index_digits=4)}.json', 'w') as outfile:
            json.dump({'shard-id': self.shard_id, 'shard-count': self.shard_count, 'successful': successful}, outfile)

    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs the conversion of a single session (single job output of Maglev Stable-Diffusion pp workflow)
        """

        self.sequence_path = sequence_path

        # Read rig json file and sensor information
        with open(sequence_path / 'rig.json', 'r') as fp:
            self.rig = json.load(fp)

        self.sensors_calibration_data = parse_rig_sensors_from_dict(self.rig)

        # Determine session-id to be processed
        self.session_id = load_maglev_session_id(self.sequence_path)
        self.logger.info(f'Converting session {self.session_id} [shard {self.shard_id + 1}/{self.shard_count}]')

        # Create output dir
        (self.output_dir / self.session_id).mkdir(parents=True, exist_ok=True)

        # Store initial shard meta 
        self.store_shard_meta(False)

        # Decode data from maglev WF
        self.decode_poses()

        self.decode_cameras()

        # Store final shard meta
        self.store_shard_meta(True)

    def decode_poses(self):
        logger = self.logger.getChild('decode_poses')
        logger.info(f'Loading poses')

        # Load timestamped poses variables
        self.global_T_rig_worlds, self.global_T_rig_world_timestamps_us = load_maglev_egomotion(
            self.sequence_path, self.sensors_calibration_data)

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
        global_target_start_timestamp_us, global_target_end_timestamp_us = self.time_bounds(
            self.global_T_rig_world_timestamps_us, self.seek_sec, self.duration_sec)
        global_range_start = np.argmax(self.global_T_rig_world_timestamps_us >= global_target_start_timestamp_us)
        global_range_end               = np.argmin(self.global_T_rig_world_timestamps_us <= global_target_end_timestamp_us) \
                                         if global_target_end_timestamp_us < self.global_T_rig_world_timestamps_us[-1] \
                                         else len(self.global_T_rig_world_timestamps_us) # full range of poses or restriction
        self.global_T_rig_worlds = self.global_T_rig_worlds[global_range_start:global_range_end]
        self.global_T_rig_world_timestamps_us = self.global_T_rig_world_timestamps_us[
            global_range_start:global_range_end]
        self.global_start_timestamp_us = self.global_T_rig_world_timestamps_us[0]
        self.global_end_timestamp_us = self.global_T_rig_world_timestamps_us[-1]

        assert self.global_start_timestamp_us >= global_target_start_timestamp_us
        assert self.global_end_timestamp_us <= global_target_end_timestamp_us

        # Apply uniform subdivision for current shard to get local pose range
        local_range, _ = uniform_subdivide_range(self.shard_id, self.shard_count, 0, len(self.global_T_rig_worlds))
        local_T_rig_world_timestamps_us = self.global_T_rig_world_timestamps_us[local_range]
        self.local_start_timestamp_us = local_T_rig_world_timestamps_us[0]
        self.local_end_timestamp_us = local_T_rig_world_timestamps_us[-1]

        assert self.local_start_timestamp_us >= self.global_start_timestamp_us
        assert self.local_end_timestamp_us <= self.global_end_timestamp_us

        # Log base pose to share it more easily with downstream teams (it's serialized also explicitly)
        with np.printoptions(floatmode='unique', linewidth=200):  # print in highest precision
            logger.info(
                f'> processed {len(local_range)} / {global_range_end - global_range_start} local / global poses, using base pose:\n{T_rig_world_base}'
            )

    def decode_cameras(self):
        logger = self.logger.getChild('decode_cameras')
        logger.info(f'Loading camera data [shard {self.shard_id + 1}/{self.shard_count}]')

        # Position interpolator to sample velocities
        position_interpolator = PositionInterpolator(self.global_T_rig_worlds, self.global_T_rig_world_timestamps_us)

        # Per-camera crop-transformation specification
        @dataclass
        class CropTransform(dataclasses_json.DataClassJsonMixin):
            crop_offset: np.ndarray = util.numpy_array_field(np.uint64)
            crop_size: np.ndarray = util.numpy_array_field(np.uint64)
            downscale_factor: int = 1

            def __post_init__(self):
                # Sanity checks
                assert self.crop_size[0] % self.downscale_factor == 0, 'height not divisible by downscale factor'
                assert self.crop_size[1] % self.downscale_factor == 0, 'width not divisible by downscale factor'

        # Specify sensors to be processed and their sensor-specific crop transformation
        CAMERAID_TO_CROPTRANSFORM = {
            'camera_front_wide_120fov':
            CropTransform(np.array((0, 0), dtype=np.int32), np.array((3840, 1920), dtype=np.int32), 5),
            'camera_cross_left_120fov':
            CropTransform(np.array((0, 0), dtype=np.int32), np.array((3840, 1920), dtype=np.int32), 5),
            'camera_cross_right_120fov':
            CropTransform(np.array((0, 0), dtype=np.int32), np.array((3840, 1920), dtype=np.int32), 5),
            'camera_rear_left_70fov':
            CropTransform(np.array((0, 0), dtype=np.int32), np.array((3840, 1920), dtype=np.int32), 5),
            'camera_rear_right_70fov':
            CropTransform(np.array((0, 0), dtype=np.int32), np.array((3840, 1920), dtype=np.int32), 5),
            'camera_rear_tele_30fov':
            CropTransform(np.array((0, 0), dtype=np.int32), np.array((3840, 1920), dtype=np.int32), 5),
        }

        # Process all camera sensors
        for camera_id, camera_rig_name in self.CAMERAID_TO_RIGNAME.items():

            # Load crop transform for current camera, skip cameras that don't have a crop-transformation
            if not (crop_transform := CAMERAID_TO_CROPTRANSFORM.get(camera_id, None)):
                logger.info(f'Skipping camera {camera_rig_name}')
                continue

            logger.info(f'Processing camera {camera_rig_name}')

            camera_type = self.CAMERAID_TO_CAMERATYPE[camera_id]

            # Load frame numbers and timestamps
            raw_frame_numbers, raw_frame_timestamps_us = load_maglev_camera_indexer_frame_meta(
                self.sequence_path / 'cameras' / camera_rig_name)

            assert len(raw_frame_numbers) == len(raw_frame_timestamps_us)

            # Map raw frame timestamps to end-of-frame timestamps respecting exposure times in middle of row
            raw_frame_timestamps_us = raw_frame_timestamps_us - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type]

            # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
            global_range_start = np.argmax(
                raw_frame_timestamps_us -
                self.CAMERATYPE_TO_ROLLINGSHUTTERDELAY_US[camera_type] >= self.global_start_timestamp_us)
            global_range_end = np.argmax(raw_frame_timestamps_us > self.global_end_timestamp_us) \
                if raw_frame_timestamps_us[-1] > self.global_end_timestamp_us else len(raw_frame_timestamps_us) # take all frames if all are within egomotion range, or determine last valid frame

            global_frame_numbers = raw_frame_numbers[global_range_start:global_range_end]
            global_frame_timestamps_us = raw_frame_timestamps_us[global_range_start:global_range_end]

            local_range, _ = uniform_subdivide_range(self.shard_id, self.shard_count, 0,
                                                     len(global_frame_timestamps_us))

            # Subsample frames to valid local ranges
            local_frame_numbers = global_frame_numbers[local_range]
            local_frame_timestamps_us = global_frame_timestamps_us[local_range]

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
            _, max_angle = compute_ftheta_parameters(np.concatenate((intrinsic, fw_poly)))

            camera_model_parameters = types.FThetaCameraModelParameters(
                intrinsic[2:4].astype(np.uint64), types.ShutterType.ROLLING_TOP_TO_BOTTOM,
                self.CAMERATYPE_TO_EXPOSURETIME_US[camera_type].item(), intrinsic[0:2], bw_poly, fw_poly,
                float(max_angle))

            # Assemble camera meta-data
            meta = {
                'T_sensor_rig': T_sensor_rig.tolist(),
                'camera_model_type': camera_model_parameters.type(),
                'camera_model_parameters': camera_model_parameters.to_dict(),
                'crop_transform': crop_transform.to_dict(),
            }

            # Prepare crop-transformation
            crop_box = np.hstack([crop_transform.crop_offset, crop_transform.crop_offset + crop_transform.crop_size])
            target_resolution = crop_transform.crop_size // crop_transform.downscale_factor

            # Load tar file containing images
            tar_file = open(self.sequence_path / 'cameras' / camera_rig_name / 'images.tar', 'rb')
            tar_index = json.load(open(self.sequence_path / 'cameras' / camera_rig_name / 'images.tar.idx.json', 'r'))

            ## Process all valid images + store in webdatasets
            with wds.ShardWriter(
                    pattern=str(self.output_dir / self.session_id /
                                (f'{camera_id}_{self.shard_id + 1}-{self.shard_count}_%d.tar')),
                    maxsize=self.wds_shard_maxsize_mib * 2**20,  # MiB -> B
                    maxcount=self.wds_shard_maxsamples,
            ) as sink:
                for continous_local_frame_index, (frame_number, frame_end_timestamp_us) in \
                    tqdm.tqdm(enumerate(zip(local_frame_numbers, local_frame_timestamps_us)), total=len(local_frame_numbers)):

                    # Determine current speed
                    speeds_km_h = position_interpolator.get_speeds_km_h(frame_end_timestamp_us).item()

                    # Filter image based on local speed if threshold is set
                    if self.min_speed_km_h and speeds_km_h < self.min_speed_km_h:
                        continue

                    # Load image file data from archive
                    file_record = tar_index[f'./{str(frame_number)}.jpeg']
                    tar_file.seek(file_record['offset_data'])
                    frame_data = types.EncodedImageData(tar_file.read(file_record['size']), 'jpeg')

                    # Decode full source image and apply crop-transformation
                    img_croptransformed = frame_data.get_decoded_image().resize(
                        target_resolution, box=crop_box.astype(np.float32),
                        resample=PILImage.LANCZOS)  # crop and downsample ROI

                    # Store image and meta data to webdataset shard (add frame-specific data to static meta-data)
                    sample = {
                        "__key__": util.padded_index_string(continous_local_frame_index),
                        "cropped.jpeg": img_croptransformed,
                        "json": meta | {
                            'speeds_km_h': speeds_km_h
                        },
                    }
                    sink.write(sample)

                logger.info(f'> processed {len(local_frame_numbers)} local'
                            f' images [shard {self.shard_id + 1}/{self.shard_count}]')

        logger.info(f'> processed {len(CAMERAID_TO_CROPTRANSFORM)} cameras')
