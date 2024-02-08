# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import base64
import logging
import os
import hashlib
import csv
import re

from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import tqdm
import numpy as np
import pyarrow.parquet as pq

from PIL import Image
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
from multimethod import multimethod

from ncore.impl.common.common import HalfClosedInterval, MaskImage, PoseInterpolator, load_jsonl
from ncore.impl.av_utils import isWithin3DBBox
from ncore.impl.common.transformations import euler_2_so3, lat_lng_alt_2_ecef, se3_inverse, transform_bbox
from ncore.impl.data.types import FrameLabel3, BBox3, LabelSource, TrackLabel, DynamicFlagState


def extract_pose(data, earth_model="WGS84"):
    """Extract the pose of the SDC

    Args:
        data (dict): pose data
    Out:
        (np.array): Transformation from SDC to ECEF coordinate system [m,4,4]
    """

    lat_lng_alt = np.array(
        [
            data["lat_lng_alt"]["latitude_degrees"],
            data["lat_lng_alt"]["longitude_degrees"],
            data["lat_lng_alt"]["altitude_meters"],
        ]
    ).reshape(-1, 3)

    rot_axis = np.array([data["axis_angle"]["x"], data["axis_angle"]["y"], data["axis_angle"]["z"]]).reshape(-1, 3)

    rot_angle = np.array(data["axis_angle"]["angle_degrees"]).reshape(-1, 1)

    return lat_lng_alt_2_ecef(lat_lng_alt, rot_axis, rot_angle, earth_model)[0]


def get_sensor_to_sensor_flu(sensor):
    """Compute a rotation transformation matrix that rotates sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.ndarray: the resulting rotation matrix.
    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        rot = np.eye(4, dtype=np.float32)

    return np.asarray(rot, dtype=np.float32)


def parse_rig_sensors_from_dict(rig) -> dict[str, dict]:
    """Parses the provided rig dictionary into a dictionary indexed by sensor name.

    Args:
        rig (Dict): Complete rig file as a dictionary.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Parse the properties from the rig file
    sensors = rig["rig"]["sensors"]

    sensors_dict = {sensor["name"]: sensor for sensor in sensors}
    return sensors_dict


def parse_rig_sensors_from_file(rig_fp):
    """Parses the provided rig file into a dictionary indexed by sensor name.

    Args:
        rig_fp (str): Filepath to rig file.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Read the json file
    with open(rig_fp, "r") as fp:
        rig = json.load(fp)

    return parse_rig_sensors_from_dict(rig)


def sensor_to_rig(sensor) -> Optional[np.ndarray]:

    sensor_name = sensor["name"]
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)

    if "nominalSensor2Rig_FLU" not in sensor:
        # Some sensors (like CAN sensors) don't have an associated sensorToRig
        return None

    nominal_T = sensor["nominalSensor2Rig_FLU"]["t"]
    nominal_R = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]

    correction_T = np.zeros(3, dtype=np.float32)
    correction_R = np.zeros(3, dtype=np.float32)

    if "correction_rig_T" in sensor.keys():
        correction_T = sensor["correction_rig_T"]

    if "correction_sensor_R_FLU" in sensor.keys():
        assert "roll-pitch-yaw" in sensor["correction_sensor_R_FLU"].keys(), str(sensor["correction_sensor_R_FLU"])
        correction_R = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]

    nominal_R = euler_2_so3(nominal_R)
    correction_R = euler_2_so3(correction_R)

    R = nominal_R @ correction_R
    T = np.array(nominal_T, dtype=np.float32) + np.array(correction_T, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T

    sensor_to_rig = transform @ sensor_to_FLU

    return sensor_to_rig


def camera_intrinsic_parameters(sensor: dict, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Parses the provided rig-style camera sensor dictionary into FTheta camera intrinsic parameters.

    Note: Only supporting FTheta 'pixeldistance-to-angle' ("bw-poly") polynomials up to 5th order (six coefficients)

    Args:
        sensor: the dictionary of the sensor parameters read from the rig file
        logger: if provided, the logger to issue warnings in (e.g., on not supported coefficients)
    Returns:
        intrinsic: array of FTheta intrinsics [cx, cy, width, height, [bwpoly]]
    """

    assert sensor["properties"]["Model"] == "ftheta", "unsupported camera model (only supporting FTheta)"

    cx = float(sensor["properties"]["cx"])
    cy = float(sensor["properties"]["cy"])
    width = float(sensor["properties"]["width"])
    height = float(sensor["properties"]["height"])

    if "bw-poly" in sensor["properties"]:
        # Legacy 4th order backwards-polynomial
        bwpoly = [np.float32(val) for val in sensor["properties"]["bw-poly"].split()]
        assert len(bwpoly) == 5, "expecting 4th-order coefficients for 'bw-poly / 'pixeldistance-to-angle' polynomial"
    elif "polynomial" in sensor["properties"]:
        # Two-way forward / backward polynomial encoding
        assert (
            sensor["properties"]["polynomial-type"] == "pixeldistance-to-angle"
        ), f"currently only supporting 'pixeldistance-to-angle' polynomial type, received '{sensor['properties']['polynomial-type']}'"

        bwpoly = [np.float32(val) for val in sensor["properties"]["polynomial"].split()]

        if len(bwpoly) - 1 > 5:
            # > 5th-order polynomials are currently not supported in the software-stack - it is not valid to simply drop higher-order terms, so exit with error for now.
            # If required in the future, a possible workaround is to "fit" a lower-order polynomial to evaluations of the higher-order inputs, but could introduce
            # too much approximation errors.
            raise ValueError(f"> encountered > 5th-order distortion polynomial for camera '{sensor['name']}'")

        # Affine term is currently not supported, issue a warning if it differs from identity
        # TODO: properly incorporate c,d,e coefficients of affine term [c, d; e, 1] into software stack (internal camera models + NGP)
        A = np.matrix(
            [
                [np.float32(sensor["properties"].get("c", 1.0)), np.float32(sensor["properties"].get("d", 0.0))],
                [np.float32(sensor["properties"].get("e", 0.0)), np.float32(1.0)],
            ]
        )

        if (A != np.identity(2, dtype=np.float32)).any():
            if logger:
                logger.warn(
                    f"> *not* considering non-identity affine term '{A}' for '{sensor['name']}' - parsed model might be inaccurate"
                )

    else:
        raise ValueError("unsupported distortion polynomial type")

    intrinsic = [cx, cy, width, height] + bwpoly

    return np.array(intrinsic, dtype=np.float32)


def vehicle_bbox(rig: dict) -> np.ndarray:
    """Parses the vehicle's bounding-box from the 'vehicle' property
        of a rig and converts it into NCORE bbox conventions.

    Args:
        rig: A parsed rig json file
    Returns:
        bbox: The vehicles bounding-box represented in the rig frame
    """

    body = rig["rig"]["vehicle"]["value"]["body"]

    bbox_position = np.array(
        body["boundingBoxPosition"], dtype=np.float32
    )  # defined as 'midpoint of rear bottom edge' in rig frame

    length = body["length"]
    width = body["width"]
    height = body["height"]

    # only offsets in x/z are required to determine centroid, as bbox_position is already centered laterally
    centroid = bbox_position + np.array([length / 2, 0.0, height / 2], dtype=np.float32)
    dimensions = np.array([length, width, height], dtype=np.float32)
    orientation = np.zeros(3, dtype=np.float32)  # vehicle bbox is aligned with the rig, i.e., is an axis-aligned bbox

    return np.hstack([centroid, dimensions, orientation])


def camera_car_mask(sensor, scale_to_source_resolution=True):
    """Parses a camera car-mask image from a rig-style camera sensor dictionary.

       Supports car masks encoded in
         - 'data/rle16-base64' (base64 string encoding of a 16bit RLE compression)
       formats

    Args:
        sensor (Dict): the dictionary of the camera sensor read from a rig file.
        scale_to_source_resolution (Bool): whether to re-scale the mask to the original sensor resolution (default = True)
    Returns:
        car_mask_image (MaskImage): mask image encoding the ego-vehicle pixels
    """

    ## Make sure this is a camera sensor that has an associated car-mask
    assert "protocol" in sensor and sensor["protocol"].startswith("camera"), "provided sensor is not a camera sensor"
    assert "car-mask" in sensor, "provided camera sensor is missing an associated 'car-mask'"

    ## Make sure we know how to load the data
    car_mask_obj = sensor["car-mask"]
    assert "data/rle16-base64" in car_mask_obj, "unsupported car-mask encoding"
    assert "resolution" in car_mask_obj, "car-mask is missing image resolution"

    ## Load the data
    resolution = np.array(car_mask_obj["resolution"])
    rle16_base64 = car_mask_obj["data/rle16-base64"]

    # Decode base64 part
    rle16 = np.frombuffer(base64.b64decode(rle16_base64), dtype=np.uint8)

    # Decode rle-16 compression
    RLE_COUNT_BYTES = 16 // 8
    RLE_COUNT_TYPE = np.uint16
    assert len(rle16) % (RLE_COUNT_BYTES + 1) == 0, "decoded base64 string is not a valid rle16 compression"

    # allocate raw output buffer
    decoded_rle16 = np.empty(resolution[0] * resolution[1], dtype=np.uint8)

    # undo run-length encoding
    with np.nditer(rle16) as input_it:
        decoded_rle16_position = 0
        count_buffer = np.empty(RLE_COUNT_BYTES, dtype=np.uint8)
        while not input_it.finished:
            # parse count
            for i in range(RLE_COUNT_BYTES):
                count_buffer[i] = input_it.value
                input_it.iternext()

            count = count_buffer.view(dtype=RLE_COUNT_TYPE)[0]

            # parse value
            value = input_it.value
            input_it.iternext()

            # output 'value' for count times
            decoded_rle16[decoded_rle16_position : decoded_rle16_position + count] = value
            decoded_rle16_position += count

        assert len(decoded_rle16) == decoded_rle16_position, "RLE decoding "
        "resulted in non-consistent number of elements relative to expected buffer size"

    # binary array in input mask resolution (True indicates pixels observing the ego-vehicle)
    car_mask = decoded_rle16.reshape(resolution[1], resolution[0]) == 0

    if scale_to_source_resolution:
        # rescale to original resolution (DW makes sure that the downscaled mask
        # is an even subsampling of the original camera resolution)
        width, height = camera_intrinsic_parameters(sensor, None)[[2, 3]].astype(
            np.int32
        )  # load original sensor resolution

        car_mask_image = Image.fromarray(car_mask).resize(
            (width, height)
        )  # convert to image and perform nearest-neighor resampling

        car_mask = np.array(car_mask_image)  # convert back to binary array, now in original sensor resolution

    # convert to mask image
    car_mask_image = MaskImage(car_mask.shape, initial_masks=[(car_mask, MaskImage.MaskType.EGO)])

    return car_mask_image


class LabelProcessor:
    """Base class providing facilities to parse / process NV labels into common NCORE format (V3)"""

    LABELCLASS_STRING_TO_LABELCLASS_ID: dict[str, int] = {
        "unknown": 0,
        "automobile": 1,
        "pedestrian": 2,
        "sign": 3,
        "CYCLIST": 4,
        "heavy_truck": 5,
        "bus": 6,
        "other_vehicle": 7,
        "motorcycle": 8,
        "motorcycle_with_rider": 9,
        "person": 10,
        "rider": 11,
        "bicycle_with_rider": 12,
        "bicycle": 13,
        "stroller": 14,
        "person_group": 15,
        "unclassifiable_vehicle": 16,
        "cycle": 17,
        "trailer": 18,
        "protruding_object": 19,
        "animal": 20,
        "train_or_tram_car": 21,
    }

    LABELCLASS_ID_TO_LABELCLASS_STRING: dict[int, str] = {v: k for k, v in LABELCLASS_STRING_TO_LABELCLASS_ID.items()}

    LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC: set[str] = set(
        [
            "pedestrian",
            "stroller",
            "person",
            "person_group",
            "rider",
            "bicycle_with_rider",
            "bicycle",
            "CYCLIST",
            "motorcycle",
            "motorcycle_with_rider",
            "cycle",
        ]
    )
    LABEL_STRINGS_UNCONDITIONALLY_STATIC: set[str] = set(["unknown", "sign"])

    # Label BBOX padding distance (in meters) to enlarge bounding boxes for per-point dynamic-flag assignment
    LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS = 3.0

    # TODO: check if this user-defined velocity threshold makes sense
    GLOBAL_SPEED_DYNAMIC_THRESHOLD = 1.0 / 3.6

    # Minimal label centroid to rig distance (skip potential self-classifications)
    MIN_CENTROID_RIG_DISTANCE_METER = 3.0

    # Use conservative +- margin time to maintain labels at frame boundaries if filtering for time-ranges
    TIME_RANGE_MARGIN_US = int(0.1 * 1e6)  # 0.1sec in usec

    @classmethod
    def parse(
        cls,
        labels_path: str,
        sensor_metas: dict[str, LidarIndexerMeta],  # parsed per sensor meta data
        T_sensor_rigs: dict[str, np.ndarray],  # per-sensor T_sensor_rig transformations
        T_rig_world_timestamps_us: np.ndarray,  # timestamps of rig-to-world poses
        T_rig_worlds: np.ndarray,  # rig-to-world poses
        source: LabelSource,
        min_centroid_rig_distance: float = MIN_CENTROID_RIG_DISTANCE_METER,
    ) -> Tuple[dict[str, TrackLabel], dict[str, dict[int, list[FrameLabel3]]], dict[str, bool]]:
        """Parses a labels file for label tracks and per-frame labels.

        Supports labels in
            - .parquet (lidar-associated autolabel 4D cuboids)
        formats

        Args:
            labels_path: path to labels file
            sensor_meta_files: per-sensor meta-files to obtain frame_number -> frame-timestamp infos
            start_timestamp_us / end_timestamp_us: start / end timestamp bounds
        Returns:
            track_labels: all tracked labels
            frame_labels: all per-frame labels for each sensor
            track_global_dynamic_flag: all "global" per-track dynamic flags
        """

        # Initialize labels struct for current lidar
        track_labels: dict[str, TrackLabel] = {}  # {TrackLabel} in track_labels[track_id]
        frame_labels: dict[
            str, dict[int, list[FrameLabel3]]
        ] = {}  # [FrameLabel3] in frame_labels[<sensor-id>][frame_timestamp_us]

        # Load per-frame timestamps for each sensor to associate the labels with frame IDs (given by end-of-frame timestamps)
        # (using a dict to error out on missing per-frame timestamps)
        sensor_frame_timestamps: dict[str, dict[int, int]] = {
            sensor_id: {
                int(frame_number): int(frame_endtime_us)
                for frame_number, frame_endtime_us in zip(sensor_meta.frame_numbers, sensor_meta.frame_endtimes_us)
            }
            for sensor_id, sensor_meta in sensor_metas.items()
        }

        # Initialize pose interpolator
        pose_interpolator = PoseInterpolator(T_rig_worlds, T_rig_world_timestamps_us)

        # TODO: add format selection once multiple different formats need to be supported (not required currently for single-format)

        # Load parquet file and convert to pandas dataframe
        label_data = pq.ParquetDataset(labels_path).read().to_pandas()

        # WAR: H8.1 RWD cuboid tracks seems to contain invalid labels -> drop rows that contain NaN + issue warning if data was dropped
        original_num_labels = len(label_data)
        label_data.dropna(inplace=True)
        if diff_rows := (original_num_labels - len(label_data)):
            logging.warn(
                f"Dropped {diff_rows} rows of cuboid labels due to NaN - resulting tracks might be wrong / incomplete"
            )
        del original_num_labels

        # Fix float -> integer datatypes of track IDs / timestamps
        label_data = label_data.astype({"gt_trackline_id": "int64", "timestamp": "int64"})

        ## Conservatively pre-filter data range based on time bounds, to speed up processing in case of restricted seek / duration data ranges

        # all of the rows with timestamp <= end-timestamp + + margin-time
        label_data = label_data[label_data["timestamp"].le(T_rig_world_timestamps_us[-1] + cls.TIME_RANGE_MARGIN_US)]

        # all of the rows with start-timestamp - margin-time <= timestamp <= end-timestamp + margin-time
        label_data = label_data[label_data["timestamp"].ge(T_rig_world_timestamps_us[0] - cls.TIME_RANGE_MARGIN_US)]

        # Restrict to columns of interest to reduce memory usage (around 70% data reduction)
        label_data = label_data[
            [
                "label_name",
                "sensor_name",
                "gt_trackline_id",
                "timestamp",
                "label_id",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "dim_x",
                "dim_y",
                "dim_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "confidence",
                "frame_number",  # used to associated labels with source-frames
            ]
        ]
        # Note: more recent data should contain a 'timestamp_origin',
        # which can be used to distinguish between 'spin-end' or 'per-cuboid' and whether
        # motion-compensation to the spin-end timestamp is required - do this unconditionally for now

        # Sort labels by timestamp to guarantee timestamp-sorted tracks
        label_data.sort_values(by=["timestamp"], inplace=True)

        for row in tqdm.tqdm(label_data.itertuples(), total=len(label_data)):
            if not row.label_name in cls.LABELCLASS_STRING_TO_LABELCLASS_ID.keys():
                logging.warn(f"> unhandled label class {row.label_name}")
                continue

            # load relevant label data
            sensor_id = row.sensor_name
            track_id = hashlib.sha1(str(row.gt_trackline_id).encode()).hexdigest()
            label_frame_number = int(row.frame_number)
            label_timestamp_us = int(row.timestamp)
            label_class = row.label_name
            label_frame_timestamp_us = sensor_frame_timestamps[sensor_id][label_frame_number]

            # make sure we can interpolate sensor poses for the relevant timestamps
            if (
                (label_timestamp_us < T_rig_world_timestamps_us[0])
                or (label_frame_timestamp_us < T_rig_world_timestamps_us[0])
                or (label_timestamp_us > T_rig_world_timestamps_us[-1])
                or (label_frame_timestamp_us > T_rig_world_timestamps_us[-1])
            ):
                continue

            # load bounding-box geometry (represented in sensor's frame at label-time)
            bbox_labeltime = BBox3(
                centroid=(row.centroid_x, row.centroid_y, row.centroid_z),
                dim=(row.dim_x, row.dim_y, row.dim_z),
                rot=(
                    row.rot_x,
                    row.rot_y,
                    row.rot_z,
                ),
            )

            # apply motion-compensation transformation to bounding box (transform bbox from sensor at label-time to sensor at frame-time)
            T_rig_labeltime_world, T_rig_frametime_world = pose_interpolator.interpolate_to_timestamps(
                [label_timestamp_us, label_frame_timestamp_us]
            )
            T_sensor_labeltime_world = T_rig_labeltime_world @ T_sensor_rigs[sensor_id]
            T_sensor_frametime_world = T_rig_frametime_world @ T_sensor_rigs[sensor_id]
            T_sensor_labeltime_sensor_frametime = se3_inverse(T_sensor_frametime_world) @ T_sensor_labeltime_world

            # bbox in sensor at frame-time
            bbox_frametime = BBox3.from_array(
                transform_bbox(bbox_labeltime.to_array(), T_sensor_labeltime_sensor_frametime)
            )

            # skip label if its centroid is too close to the rig
            if (
                np.linalg.norm(transform_bbox(bbox_frametime.to_array(), T_sensor_rigs[sensor_id])[:3])
                < min_centroid_rig_distance
            ):
                continue

            # this is assuming velocity is not relative to the local sensor motion, but w.r.t. fixed scene / world
            global_speed = float(np.linalg.norm([row.velocity_x, row.velocity_y, row.velocity_z]))

            # store frame label data
            if sensor_id not in frame_labels:
                frame_labels[sensor_id] = {}

            if label_frame_timestamp_us not in frame_labels[sensor_id]:
                frame_labels[sensor_id][label_frame_timestamp_us] = []

            frame_labels[sensor_id][label_frame_timestamp_us].append(
                FrameLabel3(
                    label_id=row.label_id,
                    track_id=track_id,
                    label_class=label_class,
                    global_speed=global_speed,
                    confidence=row.confidence,
                    source=source,
                    timestamp_us=label_timestamp_us,
                    bbox3=bbox_frametime,
                )
            )

            # store track label data
            if track_id not in track_labels:
                # Instantiate new track
                track_labels[track_id] = TrackLabel(sensors={})

            if sensor_id not in track_labels[track_id].sensors:
                # Instantiate new sensor observing this track
                track_labels[track_id].sensors[sensor_id] = []

            # append frame timestamp into *sorted* list (rows are processed sorted by timestamp)
            track_labels[track_id].sensors[sensor_id].append(label_frame_timestamp_us)

        return track_labels, frame_labels, cls.track_global_dynamic_flag(frame_labels)

    @staticmethod
    def track_global_dynamic_flag(
        frame_labels: dict[str, dict[int, list[FrameLabel3]]],
        # Static + defaulted parameters: allow reusing logic externally with different parameters (e.g., in tools / other data-converters)
        label_strings_unconditionally_dynamic: set[str] = LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC,
        label_strings_unconditionally_static: set[str] = LABEL_STRINGS_UNCONDITIONALLY_STATIC,
        global_speed_dynamic_threshold: float = GLOBAL_SPEED_DYNAMIC_THRESHOLD,
    ) -> dict[str, bool]:
        """Computes global per-track dynamic flag states (to be used, e.g., for lidar-point dynamic flag assignment)"""

        track_global_dynamic_flag: dict[str, bool] = {}  # bool in dynamic_tracks[track_id]

        # Load overwrites from environment variable NCORE_LABEL_TRACKIDS_FORCE_STATIC
        # in the format NCORE_LABEL_TRACKIDS_FORCE_STATIC='0286dd552c9bea9a69ecb3759e7b94777635514b 0716d9708d321ffb6a00818614779e779925365c' (white-space separated IDs)
        trackids_force_static = set([int(id) for id in os.environ.get("NCORE_LABEL_TRACKIDS_FORCE_STATIC", "").split()])

        for sensor_id in frame_labels:
            for label_frame_timestamp_us in frame_labels[sensor_id]:
                for frame_label in frame_labels[sensor_id][label_frame_timestamp_us]:
                    track_id, label_class, global_speed = (
                        frame_label.track_id,
                        frame_label.label_class,
                        frame_label.global_speed,
                    )

                    if track_id not in track_global_dynamic_flag:
                        track_global_dynamic_flag[track_id] = (
                            True if frame_label.label_class in label_strings_unconditionally_dynamic else False
                        )

                    if (
                        label_class not in label_strings_unconditionally_static
                        and global_speed >= global_speed_dynamic_threshold
                    ):
                        track_global_dynamic_flag[track_id] = True

                    if track_id in trackids_force_static:
                        logging.debug(
                            f"> forcing track_id={track_id} to be static (timestamp={label_frame_timestamp_us}, estimated global_speed={global_speed})"
                        )
                        track_global_dynamic_flag[track_id] = False

        return track_global_dynamic_flag

    @staticmethod
    def lidar_dynamic_flag(
        sensor_id: str,  # sensor id
        xyz: np.ndarray,  # points in sensor frame
        frame_timestamp_us: int,
        frame_labels: dict[str, dict[int, list[FrameLabel3]]],
        track_global_dynamic_flag: dict[str, bool],
        # Static + defaulted parameters: allow reusing logic externally with different parameters (e.g., in tools / other data-converters)
        lidar_dynamic_flag_bbox_padding_meters=LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS,
    ) -> Tuple[np.ndarray, list[FrameLabel3]]:
        """Computes per-point lidar dynamic flag by intersecting frame-associated bounding boxes of dynamic objects"""

        assert xyz.shape[1] == 3, "wrong point cloud shape"

        point_count = xyz.shape[0]

        # Initialize dynamic flag
        dynamic_flag: np.ndarray = np.full(
            point_count,
            # initialize dynamic_flag to -1 if there are no labels at all
            DynamicFlagState.STATIC.value if len(frame_labels) else DynamicFlagState.NOT_AVAILABLE.value,
            dtype=np.int8,
        )  # N x 1

        # Incorporate labels, if available
        current_frame_labels: list[FrameLabel3] = frame_labels.get(sensor_id, {}).get(
            frame_timestamp_us, []
        )  # returns empty dict if no annotations available for this frame

        # Use the bounding boxes to remove dynamic objects / set dynamic flag
        for frame_label in current_frame_labels:
            # If the object is classified to be dynamic update the points that fall in that bounding box
            if track_global_dynamic_flag[frame_label.track_id]:
                bbox = frame_label.bbox3.to_array()
                # enlarge the bounding box for the check *only*
                bbox[3:6] += lidar_dynamic_flag_bbox_padding_meters  # TODO: make sure this parameter is tuned sensibly
                dynamic_flag[isWithin3DBBox(xyz, bbox.reshape(1, -1))] = DynamicFlagState.DYNAMIC.value

        return dynamic_flag, current_frame_labels


def eval_polynomial(xs: np.ndarray, coeffs, use_horner=True):
    """Evaluates polynomial coeffs [,D] at given points [N,1]"""
    ret = np.zeros((len(xs), 1), dtype=xs.dtype)

    if not use_horner:
        for k, coeff in enumerate(coeffs):
            ret += coeff * xs**k
    else:
        for coeff in reversed(coeffs):
            ret = ret * xs + coeff

    return ret


def pixel_2_camera_ray(pixel_coords: np.ndarray, intrinsic: np.ndarray, camera_model: str):
    """Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (np.array): pixel coordinates of the selected points [n,2]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (np.array): rays in the camera coordinate system [n,3]
    """

    camera_rays = np.ones((pixel_coords.shape[0], 3))

    if camera_model == "pinhole":
        camera_rays[:, 0] = (pixel_coords[:, 0] + 0.5 - intrinsic[2]) / intrinsic[0]
        camera_rays[:, 1] = (pixel_coords[:, 1] + 0.5 - intrinsic[5]) / intrinsic[4]

    elif camera_model == "f_theta":
        pixel_offsets = np.ones((pixel_coords.shape[0], 2))
        pixel_offsets[:, 0] = pixel_coords[:, 0] - intrinsic[0]
        pixel_offsets[:, 1] = pixel_coords[:, 1] - intrinsic[1]

        pixel_norms = np.linalg.norm(pixel_offsets, axis=1, keepdims=True)

        alphas = eval_polynomial(pixel_norms, intrinsic[4:])  # evaluate bw_poly
        camera_rays[:, 0:1] = (np.sin(alphas) * pixel_offsets[:, 0:1]) / pixel_norms
        camera_rays[:, 1:2] = (np.sin(alphas) * pixel_offsets[:, 1:2]) / pixel_norms
        camera_rays[:, 2:3] = np.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze()
        camera_rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

    return camera_rays


def compute_fw_polynomial(intrinsic):

    img_width = intrinsic[2]
    img_height = intrinsic[3]
    cxcy = np.array(intrinsic[0:2])

    max_value = 0.0
    value = np.linalg.norm(np.asarray([0.0, 0.0], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([0.0, img_height], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([img_width, 0.0], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([img_width, img_height], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)

    SAMPLE_COUNT = 500
    samples_x = []
    samples_b = []
    step = max_value / SAMPLE_COUNT
    x = step

    for _ in range(0, SAMPLE_COUNT):
        p = np.asarray([cxcy[0] + x, cxcy[1]], dtype=np.float64).reshape(-1, 2)
        ray = pixel_2_camera_ray(p, intrinsic, "f_theta")
        xy_norm = np.linalg.norm(ray[0, :2])
        theta = np.arctan2(float(xy_norm), float(ray[0, 2]))
        samples_x.append(theta)
        samples_b.append(float(x))
        x += step

    x = np.asarray(samples_x, dtype=np.float64)
    y = np.asarray(samples_b, dtype=np.float64)

    def f4(x, b, x1, x2, x3, x4):
        """4th degree polynomial."""
        return b + x1 * x + x2 * (x**2) + x3 * (x**3) + x4 * (x**4)

    def f5(x, b, x1, x2, x3, x4, x5):
        """5th degree polynomial."""
        return b + x1 * x + x2 * (x**2) + x3 * (x**3) + x4 * (x**4) + x5 * (x**5)

    match bw_poly_degree := len(intrinsic[4:]) - 1:
        case 4:
            # Fit a 4th degree polynomial
            f = f4
            bounds = (
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
            )
        case 5:
            # Fit a 5th degree polynomial
            f = f5
            bounds = (
                [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf, np.inf],
            )
        case _:
            raise ValueError(f"Unsupported polynomial degree {bw_poly_degree}")

    # The constant in the polynomial should be zero, so add the `bounds` condition.
    coeffs, _ = curve_fit(f, x, y, bounds=bounds)

    return np.array([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)], dtype=np.float32)


def compute_ftheta_fov(intrinsic):
    """Computes the FOV of this camera model."""
    max_x = intrinsic[2] - 1
    max_y = intrinsic[3] - 1

    point_left = np.asarray([0.0, intrinsic[1]]).reshape(-1, 2)
    point_right = np.asarray([max_x, intrinsic[1]]).reshape(-1, 2)
    point_top = np.asarray([intrinsic[0], 0.0]).reshape(-1, 2)
    point_bottom = np.asarray([intrinsic[0], max_y]).reshape(-1, 2)

    fov_left = _get_pixel_fov(point_left, intrinsic)
    fov_right = _get_pixel_fov(point_right, intrinsic)
    fov_top = _get_pixel_fov(point_top, intrinsic)
    fov_bottom = _get_pixel_fov(point_bottom, intrinsic)

    v_fov = fov_top + fov_bottom
    hz_fov = fov_left + fov_right
    max_angle = _compute_max_angle(intrinsic)

    return v_fov, hz_fov, max_angle


def _get_pixel_fov(pt, intrinsic):
    """Gets the FOV for a given point. Used internally for FOV computation of the F-theta camera.

    Args:
        pt (np.ndarray): 2D pixel.

    Returns:
        fov (float): the FOV of the pixel.
    """
    ray = pixel_2_camera_ray(pt, intrinsic, "f_theta")
    fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
    return fov


def _compute_max_angle(intrinsic):

    p = np.asarray(
        [[0, 0], [intrinsic[2] - 1, 0], [0, intrinsic[3] - 1], [intrinsic[2] - 1, intrinsic[3] - 1]], dtype=np.float32
    )

    return max(
        max(_get_pixel_fov(p[0:1, ...], intrinsic), _get_pixel_fov(p[1:2, ...], intrinsic)),
        max(_get_pixel_fov(p[2:3, ...], intrinsic), _get_pixel_fov(p[3:4, ...], intrinsic)),
    )


def load_maglev_camera_indexer_frame_meta(camera_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns *raw* frame numbers and timestamps from meta-data of Maglev's camera-indexer"""

    frames_metadata = load_jsonl(camera_path / "meta.json")

    # WAR: Due to a bug in 'meta.json' generation it is not subsampled, but 'frames.csv' is if it exists - so incorporate
    #      this data instead along with timestamps from 'meta.json'.
    #      If 'frames.csv' doesn't exist this indicates that *all* frames were exported at full framerate,
    #      so revert to the regular meta data.
    try:
        # Figure out which frames were actually indexed
        with open(camera_path / "frames.csv", "r") as frames_file:
            raw_frame_numbers = np.array([row[0] for row in csv.reader(frames_file)], dtype=np.uint64)

        # Construct frame number to timestamp map
        frame_timestamps_map_us = {
            frame_metadata["frame_number"]: frame_metadata["timestamp"] for frame_metadata in frames_metadata
        }
        # Evaluate timestamps map for exported frames
        raw_frame_timestamps_us = np.array(
            [frame_timestamps_map_us[raw_frame_number] for raw_frame_number in raw_frame_numbers], dtype=np.uint64
        )

    except FileNotFoundError:
        # Special case: if 'frames.csv' doesn't exists the camera indexer exported all frames, so load
        #               all frames from the 'meta.json' file

        # Load all data from 'meta.json' - default case assuming 'meta.json' is subsampled to all existing frames
        raw_frame_numbers = np.array([frame_data["frame_number"] for frame_data in frames_metadata], dtype=np.uint64)
        raw_frame_timestamps_us = np.array([frame_data["timestamp"] for frame_data in frames_metadata], dtype=np.uint64)

    finally:
        return raw_frame_numbers, raw_frame_timestamps_us


@dataclass
class LidarIndexerMeta:
    frame_numbers: np.ndarray  # raw frame numbers mapping for frame file names
    frame_endtimes_us: np.ndarray  # end-of-frame times
    frames_egocompensated: bool  # if point clouds are motion-compensated
    frame_starttimes_us: Optional[np.ndarray]  # only available from lidar-exporter meta-data


def load_maglev_lidar_indexer_frame_meta(lidarpath_or_metafile: Path) -> LidarIndexerMeta:
    """Returns meta-data of Maglev's lidar-indexer variants.

    Input can either be a path to a meta.json file (CSFT-based) or a directory path containing either
    a meta.json file (CSFT-based) or 'spins.txt'/'toolConfigs.txt' files (lidar-exporter-based)"""

    # Determine meta file path to load
    if lidarpath_or_metafile.suffix == ".json":
        meta_file_path = lidarpath_or_metafile
    else:
        meta_file_path = lidarpath_or_metafile / "meta.json"

    if meta_file_path.exists():
        # Load CSFT-based indexed frame meta-data

        frames_metadata = load_jsonl(meta_file_path)

        raw_frame_numbers = np.array([frame_data["frame_number"] for frame_data in frames_metadata], dtype=np.uint64)
        raw_frame_timestamps_us = np.array([frame_data["timestamp"] for frame_data in frames_metadata], dtype=np.uint64)
        raw_frame_egocompensated = np.array(
            [frame_data["ego_compensated"] for frame_data in frames_metadata], dtype=np.bool8
        )

        # Sanity check assumptions
        assert np.all(
            raw_frame_egocompensated == raw_frame_egocompensated[0]
        ), "expecting consistent motion-compensation state"

        return LidarIndexerMeta(
            frame_numbers=raw_frame_numbers,
            frame_endtimes_us=raw_frame_timestamps_us,
            frames_egocompensated=bool(raw_frame_egocompensated[0]),
            frame_starttimes_us=None,
        )

    if (spin_file_path := lidarpath_or_metafile / "spins.txt").exists():
        # Load lidar-exporter-based meta-data / spins file

        with open(spin_file_path, "r") as spins_file:
            rows = [row for row in csv.DictReader(spins_file)]

        raw_lidar_idx = np.array([int(row["lidarIdx"]) for row in rows], dtype=np.uint64)
        raw_spin_idx = np.array([int(row["spinIndex"]) for row in rows], dtype=np.uint64)
        raw_start_time_us = np.array([int(row["startTime"]) for row in rows], dtype=np.uint64)
        raw_end_time_us = np.array([int(row["endTime"]) for row in rows], dtype=np.uint64)
        raw_primary_spin_idx = np.array([int(row["primarySpinIndex"]) for row in rows], dtype=np.uint64)

        # Determine motion-compensation property from tool configuration
        frames_egocompensated: None | bool = None
        if (toolconfig_path := lidarpath_or_metafile / "toolConfigs.txt").exists():
            with open(toolconfig_path, "r") as toolconfig_file:
                if state := re.search(r"--motionCompensate=(\d)", toolconfig_file.read()):
                    frames_egocompensated = state.group(1) == "1"
        if (toolconfig_path := lidarpath_or_metafile / "toolConfigs.json").exists():
            with open(toolconfig_path, "r") as toolconfig_file:
                frames_egocompensated = json.load(toolconfig_file)["configs"]["motionCompEnabled"] is not None

        assert (
            frames_egocompensated is not None
        ), "Can't determine motion-compensation state from lidar exporter tool-config"

        # Sanity check assumptions
        assert np.all(raw_lidar_idx == raw_lidar_idx[0]), "Expecting consistent single-lidar data"
        assert np.all(
            raw_spin_idx == raw_primary_spin_idx
        ), "Expecting spin indices to be consistent with primary spins"

        return LidarIndexerMeta(
            frame_numbers=raw_spin_idx,
            frame_starttimes_us=raw_start_time_us,
            frame_endtimes_us=raw_end_time_us,
            frames_egocompensated=frames_egocompensated,
        )

    raise ValueError("No viable lidar-indexer meta-data found")


@dataclass
class MaglevSegmentID:
    """Tracks segment-specific data"""

    segment_id: str
    segment_start_timestamp_us: int
    segment_end_timestamp_us: int


@dataclass
class MaglevSequenceID:
    """Tracks maglev dataset-specific data (source session ID and optional segment information)"""

    session_id: str  # source-session ID, always present
    segment_id: MaglevSegmentID | None  # only set if sequence is a segment (~ time-restriction of a session)

    def get_sequence_id(self) -> str:
        """Returns sequence ID - either <session-id> (for full sequences), or <session-id>[<start-time>-<end-time>] (for restricted sequences)"""

        if (segment_id := self.segment_id) is not None:
            return f"{self.session_id}[{segment_id.segment_start_timestamp_us}-{segment_id.segment_end_timestamp_us}]"

        return self.session_id


def load_maglev_sequence_id(sequence_path: Path) -> MaglevSequenceID:
    """Loads sequence-id in a trustable way"""

    session_data_path = Path(sequence_path) / "session_data"
    assert session_data_path.exists(), f"{session_data_path} doesn't exist"

    # If this is a virtual clip, use the clips as segment-id
    if (vclip_symlink_map_path := session_data_path / "vclip_symlink_map.conf").exists():
        with open(vclip_symlink_map_path, "r") as fp:
            # symlink map has the form
            # <SOURCE-SESSION_ID>_<START-TS>_<END-TS> <SEGMENT-ID> <START-TS> <END-TS> <ASSET_URL>
            # and we use <CLIP-ID> for the current "virtual" session-ID
            s = fp.read().split()
            assert len(s) == 5, ValueError(
                "Unable to parse vclip_symlink_map - unable to determine trustable sequence id"
            )

            match = re.search(r"(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})_(\d+)_(\d+)", s[0])
            if match:
                assert match[2] == s[2]  # start time
                assert match[3] == s[3]  # end time
                return MaglevSequenceID(
                    session_id=match[1],
                    segment_id=MaglevSegmentID(
                        segment_id=s[1], segment_start_timestamp_us=int(s[2]), segment_end_timestamp_us=int(s[3])
                    ),
                )
            else:
                raise ValueError("Parsing vclip_symlink_map data failed - unable to determine trustable sequence id")

    # Parse session-id as segment-id

    # Note: session_id in loaded rig meta might not reflect the actual current session due to bugs
    #       in the rig generation, prefer loading correct ID from session-data directly

    # Find `aux_info` from session-data
    aux_infos = list(session_data_path.glob("**/aux_info"))
    assert len(aux_infos), f"no 'aux_info' found in {session_data_path}"

    # Parse first `aux_info` (session-id is the same in all of them)
    with open(aux_infos[0], "r") as fp:
        match = re.search(r"uuid: (\w{8}-\w{4}-\w{4}-\w{4}-\w{12})", fp.read())
        if match:
            return MaglevSequenceID(session_id=match[1], segment_id=None)
        else:
            raise ValueError("Unable to determine trustable session_id")


@multimethod
def load_maglev_egomotion(
    sequence_path: Path, sensors_calibration_data: dict[str, dict], egomotion_file_overwrite: Optional[Path] = None
) -> Tuple[list[np.ndarray], list[int], str]:
    """Parse a maglev-based egomotion data into timestamped global T_rig_worlds

    The NV maglev 'egomotion.json' / 'egomotion.jsonl' format has gone through a couple of iterations,
    but we still support all "flavours" in a backwards compatible-way in NCore.
    The last major iteration is discussed in https://jirasw.nvidia.com/browse/GTS-7657 / https://jirasw.nvidia.com/browse/GTS-7646
    and was merged to NDAS with https://git-av.nvidia.com/r/c/ndas/+/133793
    """

    # Pre-compute sensor extrinsics to compute poses of the rig frame if egomotion is represented in a sensor frame
    T_rig_sensors = {
        sensor_name: se3_inverse(T_sensor_rig)
        for sensor_name, T_sensor_rig in {
            sensor_name: sensor_to_rig(sensors_calibration_data[sensor_name])
            for sensor_name in sensors_calibration_data
        }.items()
        if T_sensor_rig is not None
    }

    # Determine egomotion source file to parse
    if egomotion_file_overwrite:
        # Use specific egomotion input unconditionally
        return load_maglev_egomotion(T_rig_sensors, egomotion_file_overwrite)

    # Use default egomotion jsonl location - this defines the data-range unconditionally
    T_rig_worlds, T_rig_world_timestamps_us, egomotion_type = load_maglev_egomotion(
        T_rig_sensors, sequence_path / "egomotion" / "egomotion.json"
    )

    # Incorporate overwrite if no explicit external egomotion file was provided and
    if (egomotion_overwrite_file := sequence_path / "egomotion" / "egomotion-overwrite.json").exists():
        overwrite_T_rig_worlds, overwrite_T_rig_world_timestamps_us, _ = load_maglev_egomotion(
            T_rig_sensors, egomotion_overwrite_file
        )

        # Assert that overwrite time-extend does cover the default range [with small slack]
        SLACK = 1 * int(1e6)  # 1sec
        default_interval = HalfClosedInterval(T_rig_world_timestamps_us[0], T_rig_world_timestamps_us[-1] + 1)
        overwrite_interval = HalfClosedInterval(
            overwrite_T_rig_world_timestamps_us[0] - SLACK, overwrite_T_rig_world_timestamps_us[-1] + 1 + SLACK
        )

        assert (T_rig_world_timestamps_us[0] in overwrite_interval) and (
            T_rig_world_timestamps_us[-1] in overwrite_interval
        ), f"egomotion overwrite timerange {overwrite_interval} incompatible with (slacked) default bounds {default_interval}"

        # apply pose overwrite in default egomotion range and make use of overwrite type
        overwrite_range = default_interval.cover_range(np.array(overwrite_T_rig_world_timestamps_us))
        T_rig_worlds = overwrite_T_rig_worlds[overwrite_range.start:overwrite_range.stop]
        T_rig_world_timestamps_us = overwrite_T_rig_world_timestamps_us[overwrite_range.start:overwrite_range.stop]
        with open(sequence_path / "egomotion" / "egomotion-overwrite-type.txt", "r") as fp:
            egomotion_type = fp.read()

    return T_rig_worlds, T_rig_world_timestamps_us, egomotion_type


@load_maglev_egomotion.register
def _(T_rig_sensors: dict[str, np.ndarray], egomotion_file: Path) -> Tuple[list[np.ndarray], list[int], str]:
    """Parse a maglev-based egomotion data into timestamped global T_rig_worlds"""

    # Variables we are parsing into
    global_T_rig_worlds = []
    global_T_rig_world_timestamps_us = []

    # Normalize sensor names (in case "decayed" names are used as input)
    def normalize_frame_name(sensor_name: str) -> str:
        """Decay ":" -> "_" for sensor/frame names normalization"""
        return sensor_name.replace(":", "_", -1)

    T_rig_sensors = {
        normalize_frame_name(sensor_name): T_rig_sensor for (sensor_name, T_rig_sensor) in T_rig_sensors.items()
    }

    # Use different parser implementations based on the egomotion file formats
    def parse_legacy_egomotion(egomotion_file: Path) -> str:
        """Parses "jsonl"-type of egomotion file"""

        # empty string egomotion_type identifier will be returned in case there are no poses at all -
        # otherwise the first pose's frame will be used as an identifier
        egomotion_type = str("")

        for egomotion_pose_entry in load_jsonl(egomotion_file):
            # Skip invalid poses
            if not egomotion_pose_entry[
                "valid"
            ]:  # this will fail / throw an exception on non-jsonl egomotion file formats
                continue

            # Note: there is additional data like lat/long and sensor-related information
            #       which could be used in the future
            # Note: make sure all poses information is represented as f64 to have sufficient
            #       precision in case poses are representing global / map-associated coordinates
            T_rig_world_timestamp_us = int(egomotion_pose_entry["timestamp"])
            T_rig_world = (
                np.asfarray(egomotion_pose_entry["pose"].split(" "), dtype=np.float64).reshape((4, 4)).transpose()
            )

            # Make sure poses represent *rigToWorld* transformations
            # (actually *rigToGlobal* as they include the base pose also - this is the case for non-identity initial poses)
            # Note: there currently seems to be an inconsistency in the egomotion indexer output - keep
            #       this verified workaround logic for now (might need to be adapted if egomotion indexer is fixed)
            if (frame_name := normalize_frame_name(egomotion_pose_entry["in_sensor_name_frame"])) == "rig":
                # Pose is for the rig frame already - nothing to transform
                pass
            elif frame_name in T_rig_sensors:
                # Convert pose in lidar frame to pose in rig frame
                T_rig_world = T_rig_world @ T_rig_sensors[frame_name]
            else:
                raise ValueError(f"Unsupported source ego frame {frame_name}")

            # Sanity check on data-type
            assert T_rig_world.dtype is np.dtype(
                "float64"
            ), "Require pose to be double-precision (to support globally aligned / map-associated)"

            global_T_rig_worlds.append(T_rig_world)
            global_T_rig_world_timestamps_us.append(T_rig_world_timestamp_us)

            if not len(egomotion_type):
                # pick first frame name as egomotion type string, as legacy format doesn't contain variant-specific information
                egomotion_type += f"egomotion-{frame_name}"

        return egomotion_type

    def parse_egomotion(egomotion_file: Path) -> str:
        """Parses "json"-type of egomotion file"""

        with open(egomotion_file, "r") as fp:
            egomotion_json = json.load(fp)

        egomotion_type = str(egomotion_json["egomotion_type"])

        for egomotion_pose_entry in (
            egomotion_json["tf_frame_world"]
            if "tf_frame_world" in egomotion_json
            else
            # fallback for current deepmap format
            egomotion_json["poses"]
        ):  # will throw for "old" json-l format
            if not egomotion_pose_entry.get("valid", True):  # entries seem to be "implicitly" valid if key is missing
                continue

            # Note: make sure all poses information is represented as f64 to have sufficient
            #       precision in case poses are representing global / map-associated coordinates
            T_rig_world_timestamp_us = int(egomotion_pose_entry["timestamp"])

            quat = (
                egomotion_pose_entry["q_xyzw"]
                if "q_xyzw" in egomotion_pose_entry
                else
                # fallback for current deepmap format
                egomotion_pose_entry["quaternion"]
            )
            t = (
                egomotion_pose_entry["t"]
                if "t" in egomotion_pose_entry
                else
                # fallback for current deepmap format
                egomotion_pose_entry["translation"]
            )

            T_rig_world = np.block(
                [
                    [
                        R.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix(),
                        np.asarray(t, dtype=np.float64)[:, np.newaxis],
                    ],
                    [np.array([0.0, 0.0, 0.0, 1.0])],
                ]
            )

            # Make sure poses represent *rigToWorld* transformations
            # (actually *rigToGlobal* as they include the base pose also - this is the case for non-identity initial poses)
            # Note: there currently seems to be an inconsistency in the egomotion indexer output - keep
            #       this verified workaround logic for now (might need to be adapted if egomotion indexer is fixed)
            if (frame_name := normalize_frame_name(egomotion_json["coordinate_frame"])) == "rig":
                # Pose is for the rig frame already - nothing to transform
                pass
            elif frame_name in T_rig_sensors:
                # Convert pose in lidar frame to pose in rig frame
                T_rig_world = T_rig_world @ T_rig_sensors[frame_name]
            else:
                raise ValueError(f"Unsupported source ego frame {frame_name}")

            global_T_rig_worlds.append(T_rig_world)
            global_T_rig_world_timestamps_us.append(T_rig_world_timestamp_us)

        return egomotion_type

    try:
        # New egomotion json file format
        egomotion_type = parse_egomotion(egomotion_file)
    except json.decoder.JSONDecodeError:
        # Old egomotion jsonl format
        egomotion_type = parse_legacy_egomotion(egomotion_file)

    assert all(
        global_T_rig_world_timestamps_us[i] < global_T_rig_world_timestamps_us[i + 1]
        for i in range(len(global_T_rig_world_timestamps_us) - 1)
    ), "pose timestamps not monotonically increasing"

    return global_T_rig_worlds, global_T_rig_world_timestamps_us, egomotion_type
