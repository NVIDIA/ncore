# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import pickle
import re
import struct
import json
import lzma
import io
import os
import time
import sys

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import numpy as np

import PIL.Image as PILImage
from scipy import spatial, interpolate
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path, "rb")
    return pickle.load(file)


def save_pkl(obj, path):
    """
    save a dictionary to a pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pc_dat(file_path: str, allow_lookup_fallback: bool = True) -> np.ndarray:
    """
    Loads binary .dat / .dat.xz files representing a 2D single-precision array.
    Serialized 2D arrays usually represent a point-clouds with columns defined as

    [x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic_flag]

    - xys_s / xyz_e: the start / end point of world rays
    - dist: the norm of the ray
    - intensity: lidar intensity response value for this point
    - dynamic_flag:
      - -1: if the information is not available,
      -  0: static
      -  1: = dynamic

    Args:
        file_path: path to .dat / .dat.xz file to load.
        allow_lookup_fallback: If enabled, will fall back to .dat.xz/.dat, resp., in case loading .dat/.dat.xz fails (for backwards-compatibility).
    Return:
        lidar_data: loaded 2D single-precision array
    """

    def load(file: Union[io.BufferedReader, lzma.LZMAFile]) -> np.ndarray:
        # The first number denotes the number of points
        n_rows, n_columns = struct.unpack("<ii", file.read(8))
        # The remaining data are floats saved in little endian
        # Columns usually contain: x_s, y_s, z_s, x_e, y_e, z_e, d, intensity, dynamic_flag
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
        return np.array(struct.unpack("<%sf" % (n_rows * n_columns), file.read()), dtype=np.float32).reshape(
            n_rows, n_columns
        )

    if file_path.endswith(".dat"):
        try:
            with open(file_path, "rb") as file:
                lidar_data = load(file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with lzma.open(file_path + ".xz", "rb") as lzma_file:
                    lidar_data = load(lzma_file)
            else:
                raise e
    elif file_path.endswith(".dat.xz"):
        try:
            with lzma.open(file_path, "rb") as lzma_file:
                lidar_data = load(lzma_file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with open(file_path.replace(".dat.xz", ".dat"), "rb") as file:
                    lidar_data = load(file)
            else:
                raise e
    else:
        raise ValueError("invalid file format provided, supporting .dat / .dat.xz files only")

    return lidar_data


def save_pc_dat(file_path: str, lidar_data: np.ndarray) -> None:
    """
    Stores binary .dat / .dat.xz file representing a 2D single-precision array, usually representing
    a point-cloud (see load_pc_dat for format description).

    Args:
        file_path: path to .dat / .dat.xz file to store
        lidar_data: 2D single-precision array to serialize
    """

    if lidar_data.dtype is not np.dtype("float32"):
        raise ValueError("expecting single-precision array as input")

    def save(file: Union[io.BufferedWriter, lzma.LZMAFile]) -> None:
        n_rows, n_columns = lidar_data.shape
        lidar_data_flat = lidar_data.flatten()

        file.write(struct.pack("<i", n_rows))
        file.write(struct.pack("<i", n_columns))
        file.write(struct.pack("<%sf" % lidar_data_flat.size, *lidar_data_flat))

    if file_path.endswith(".dat"):
        with open(file_path, "wb") as file:
            save(file)
    elif file_path.endswith(".dat.xz"):
        with lzma.open(
            file_path,
            "wb",
            # Use fastest possible compression mode which still gives acceptable compression rates
            preset=0,
        ) as lzma_file:
            save(lzma_file)
    else:
        raise ValueError("invalid file format provided, supporting .dat / .dat.xz files only")


def load_jsonl(jsonl_path: Union[str, Path]) -> List[dict]:
    """
    Loads a jsonl (json-lines) file (each line corresponds to a serialized json object) - see jsonlines.org

    Args:
        jsonl_path: json-lines file path
    Return:
        object_list: list of parsed objects
    """

    object_list = []
    with open(jsonl_path, "r") as fp:
        for line in fp:
            object_list.append(json.loads(line))

    return object_list


def save_jsonl(file_path: str, object_list: List[dict]) -> None:
    """
    Saves a list of json-serializable objects into a jsonl (json-lines) file (each line corresponds to a serialized json object) - see jsonlines.org

    Args:
        jsonl_path: json-lines output file path
        object_list: list of json-serializable objects to save
    """

    with open(file_path, "w") as fp:
        fp.writelines([json.dumps(object) + "\n" for object in object_list])


def platform_cpu_count(upper_limit: Optional[int] = None) -> int:
    """Determines CPU count in MagLev-compatible way (with an optional upper limit)"""

    # Check if we are running in a MagLev workflow and return its CPU limits, otherwise fall back to regular CPU count
    cpu_count = int(os.environ.get("WORKFLOW_CPU_LIMITS", str(os.cpu_count())))

    if upper_limit:
        cpu_count = min(upper_limit, cpu_count)

    return cpu_count


def average_camera_pose(poses):
    """
    Compute the average position of the camera
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    average_cam_position = poses[:, :3, 3].mean(0)
    pose_min, pose_max = np.min(poses[:, :3, 3], axis=0), np.max(poses[:, :3, 3], axis=0)
    extent_scene = np.max(pose_max - pose_min)

    return average_cam_position, extent_scene


class PoseInterpolator:
    """
    Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations. https://en.wikipedia.org/wiki/Slerp

    Args:
        poses (np.array): poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array): timestamps of the known poses [n]
        ts_target (np.array): timestamps for which the poses will be interpolated [m,1]
    Out:
        (np.array): interpolated poses in se3 representation [m,4,4]
    """

    def __init__(self, poses, timestamps):

        self.slerp = spatial.transform.Slerp(timestamps, R.from_matrix(poses[:, :3, :3]))
        self.f_x = interpolate.interp1d(timestamps, poses[:, 0, 3])
        self.f_y = interpolate.interp1d(timestamps, poses[:, 1, 3])
        self.f_z = interpolate.interp1d(timestamps, poses[:, 2, 3])

        self.last_row = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, -1)

    def interpolate_to_timestamps(self, ts_target):
        x_interp = self.f_x(ts_target).reshape(-1, 1, 1).astype(np.float32)
        y_interp = self.f_y(ts_target).reshape(-1, 1, 1).astype(np.float32)
        z_interp = self.f_z(ts_target).reshape(-1, 1, 1).astype(np.float32)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1, 3, 3).astype(np.float32)

        t_interp = np.concatenate([x_interp, y_interp, z_interp], axis=-2)

        return np.concatenate(
            (np.concatenate([R_interp, t_interp], axis=-1), np.tile(self.last_row, (R_interp.shape[0], 1, 1))), axis=1
        )


def get_2d_bbox_corners(bbox):

    bbox_corners = np.zeros((4, 2))

    bbox_corners[0, :] = np.array([bbox[0] - 0.5 * bbox[2], bbox[1] - 0.5 * bbox[3]]).astype(np.int32)  # TL
    bbox_corners[1, :] = np.array([bbox[0] - 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]).astype(np.int32)  # BL
    bbox_corners[2, :] = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]).astype(np.int32)  # BR
    bbox_corners[3, :] = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] - 0.5 * bbox[3]]).astype(np.int32)  # TR

    return bbox_corners


def get_3d_bbox_coords(bbox3d):
    x, y, z = bbox3d[0], bbox3d[1], bbox3d[2]
    length, width, height = bbox3d[3], bbox3d[4], bbox3d[5]
    rotation_angles = bbox3d[6:9]

    # Computes the coordinates of the bbox corners
    l2 = length / 2
    w2 = width / 2
    h2 = height / 2

    translation = np.array([x, y, z]).reshape(1, 3)

    P1 = np.array([-l2, -w2, -h2])  # BBR (back bottom right)
    P2 = np.array([-l2, -w2, h2])  # BTR (back top right)
    P3 = np.array([-l2, w2, h2])  # BTL (back top left)
    P4 = np.array([-l2, w2, -h2])  # BBL (back bottom left)
    P5 = np.array([l2, -w2, -h2])  # FBR (front bottom right)
    P6 = np.array([l2, -w2, h2])  # FTR (front top right)
    P7 = np.array([l2, w2, h2])  # FTL (front top left)
    P8 = np.array([l2, w2, -h2])  # FBL (front bottom left)

    # Get the rotation matrix from the heading angle
    rotation = R.from_euler("xyz", rotation_angles, degrees=False).as_matrix()

    corners = np.stack([P1, P2, P3, P4, P5, P6, P7, P8], axis=0)

    corners = np.matmul(rotation, corners.transpose()).transpose() + translation

    return corners


def compute_optimal_assignments(corr_2d_3d, corr_3d_2d, cameras):
    optimal_assignments = {}
    # Iterate over the cameras
    for cam in cameras:
        optimal_assignments[cam] = {}
        # Build the cost matrix
        C = np.ones((len(corr_3d_2d[cam].keys()), len(corr_2d_3d[cam].keys()))) * 200

        tmp_labels_2d = list(corr_2d_3d[cam].keys())
        tmp_labels_3d = list(corr_3d_2d[cam].keys())

        for idx_3d, label_3d in enumerate(tmp_labels_3d):
            n_observations = len(corr_3d_2d[cam][label_3d]["2d_name"])
            all_2d_labels = list(set(corr_3d_2d[cam][label_3d]["2d_name"]))

            if len(all_2d_labels) == 1 and len(corr_2d_3d[cam][all_2d_labels[0]]["name"]) == 1:
                C[idx_3d, tmp_labels_2d.index(all_2d_labels[0])] = 0.0

            else:
                # Check what the ratios of the label
                ratios_3d = []
                ratios_2d = []
                median_IoU = []
                for label in all_2d_labels:
                    n_assignments = corr_3d_2d[cam][label_3d]["2d_name"].count(label)
                    ratios_2d.append(n_assignments / corr_2d_3d[cam][label]["count"])
                    ratios_3d.append(n_assignments / n_observations)
                    median_IoU.append(
                        np.median(corr_3d_2d[cam][label_3d]["iou"][corr_3d_2d[cam][label_3d]["2d_name"].index(label)])
                    )

                    C[idx_3d, tmp_labels_2d.index(label)] = 200 - (
                        10 * n_assignments * ratios_2d[-1] * ratios_2d[-1] * median_IoU[-1]
                    )

        row_ind, col_ind = linear_sum_assignment(C)

        for i in range(len(row_ind)):
            optimal_assignments[cam][tmp_labels_3d[row_ind[i]]] = tmp_labels_2d[col_ind[i]]

    return optimal_assignments


def check_overlap(bbox_1, bbox_2):

    overlap_x = np.abs(bbox_1[0] - bbox_2[0]) <= 0.5 * (bbox_1[2] + bbox_2[2])
    overlap_y = np.abs(bbox_1[1] - bbox_2[1]) <= 0.5 * (bbox_1[3] + bbox_2[3])

    return overlap_x & overlap_y


def computer_intersection_area(bbox_1, bbox_2):

    left = np.max([bbox_1[0] - bbox_1[2] * 0.5, bbox_2[0] - bbox_2[2] * 0.5])
    right = np.min([bbox_1[0] + bbox_1[2] * 0.5, bbox_2[0] + bbox_2[2] * 0.5])
    top = np.max([bbox_1[1] - bbox_1[3] * 0.5, bbox_2[1] - bbox_2[3] * 0.5])
    bottom = np.min([bbox_1[1] + bbox_1[3] * 0.5, bbox_2[1] + bbox_2[3] * 0.5])

    return (left - right) * (top - bottom)


def compute_iou(bbox_1, bbox_2):

    # If bounding boxes do not overlap return 0.0
    if not check_overlap(bbox_1, bbox_2):
        return 0.0

    b1_area = bbox_1[2] * bbox_1[3]  # Width times height
    b2_area = bbox_2[2] * bbox_2[3]  # Width times height

    # Filter if 2D bbox is bigger than 3D (3D should always be bigger as it is projected, and 2d is not amodal)
    if b2_area > b1_area:
        return 0.0

    intersection_area = computer_intersection_area(bbox_1, bbox_2)
    union_area = b1_area + b2_area - intersection_area

    iou = intersection_area / union_area

    if np.max([np.min([iou, 1.0]), 0.0]) == 1.0:
        print("wrong")

    return np.max([np.min([iou, 1.0]), 0.0])


class MaskImage:
    """
    Image encoding *per-pixel* annotation mask types:

        - dynamic [255, 255, 255] - e.g., dynamic vehicles / objects
        - ego     [0, 255, 0] - pixels corresponding to projections of the ego vehicle

    Properties can be set using binary input images. A pixel can only have a single property assigned.
    Output images are represented using color pallets to reduce memory footprints.
    """

    class MaskType(Enum):
        """Enumerates supported mask types"""

        NONE = 0
        DYNAMIC = 1
        EGO = 2

    # Define colors of mask types
    mask_colors = {MaskType.NONE: [0, 0, 0], MaskType.DYNAMIC: [255, 255, 255], MaskType.EGO: [0, 0, 255]}
    # Initialize color pallet with all color entries (flattening all individual RBG colors into pallet)
    palette = [
        color_component
        for mask_color in [
            # note: ordering is crucial as integer color indices map to mask types
            mask_colors[MaskType.NONE],
            mask_colors[MaskType.DYNAMIC],
            mask_colors[MaskType.EGO],
        ]
        for color_component in mask_color
    ]

    def __init__(self, shape, initial_masks=None):
        """
        Initializes a MaskImage object to a given mask shape with optional initial masks
        Args:
            shape: array shape corresponding to image (height, width)
            initial_masks: if provided, an iterable of [(binary_mask, MaskType), ...] tuples to initialize the mask image with in order
        """
        # initialize empty mask array corresponding to NONE type of appropriate type
        self.mask_array = np.full(shape, MaskImage.MaskType.NONE.value, dtype=np.uint8)

        # apply initial masks if available
        if initial_masks:
            for initial_mask in initial_masks:
                self.set(*initial_mask)

    def set(self, binary_mask, mask_type):
        """
        Sets the mask type of all enabled pixels in the binary_mask to mask_type.
        Args:
            binary_mask: 2D binary array of same shape as underlying image.
            mask_type: the MaskType to set the pixels to
        """
        assert isinstance(binary_mask, np.ndarray), "expecting array as input"
        assert isinstance(mask_type, MaskImage.MaskType), "expecting MaskType as input"
        assert binary_mask.dtype is np.dtype("bool"), "expecting binary array as input"
        assert (
            binary_mask.shape == self.mask_array.shape
        ), f"invalid array resolution, expecting shape {self.mask_array.shape}"

        # set new values for masked pixels
        self.mask_array[binary_mask] = mask_type.value

    def get_image(self) -> PILImage.Image:
        """
        Returns the color-paletted mask image with all mask types set
        Returns:
            mask_image: mask image with all pixel colors set to the corresponding mask types
        """
        # convert mask array to image
        mask_image = PILImage.fromarray(self.mask_array, mode="P")

        # apply color palette
        mask_image.putpalette(self.palette)

        return mask_image


class SimpleTimer:
    """Simple Timer to track runtimes"""

    def __init__(self):
        """Starts timer immediately"""

        self.start()

    def start(self) -> None:
        """(Re-)start the timer"""

        self._start_time = time.perf_counter()

    def elapsed_sec(self, restart: bool = False) -> float:
        """Returns elapsed time (in seconds) since start, optionally restarting the timer"""

        elapsed = time.perf_counter() - self._start_time

        if restart:
            self.start()

        return elapsed


def time_bounds(timestamps_us: List[int], seek_sec: Optional[float], duration_sec: Optional[float]) -> tuple[int, int]:
    """
    Determine start and end timestamps given optional seek and duration times

    Args:
        timestamps_us : list of all available timestamps (in microseconds)
        seek_sec: Optional: if non-None, the time (in seconds)  to skip starting from the first timestamp
        duration_sec: Optional: if non-None, the total time (in seconds) between the start and end time bounds

    Return:
        start_timestamp_us: first valid timestamp in restricted bounds (in microseconds)
        end_timestamp_us: last valid timestamp in restricted bounds (in microseconds)
    """

    start_timestamp_us = int(timestamps_us[0])
    end_timestamp_us = int(timestamps_us[-1])

    if seek_sec is not None:
        assert seek_sec >= 0.0, "Require positive seek time"
        start_timestamp_us += int(seek_sec * 1e6)

    if duration_sec is not None:
        assert duration_sec > 0.0, "Require positive duration time"
        end_timestamp_us = start_timestamp_us + int(duration_sec * 1e6)

    assert start_timestamp_us < end_timestamp_us, "Arguments lead to invalid time bounds"

    return start_timestamp_us, end_timestamp_us


def uniform_subdivide_range(
    subdiv_id: int, subdiv_count: int, range_start: int, range_end: int
) -> Tuple[np.ndarray, np.int64]:
    """
    Splits the index range range_start:range_end into (approximately) uniform intervals
    based on the requested number of subvisions.

    Args:
        subdiv_id (int): Subdivision id, with subdiv_id < subdiv_count
        subdiv_count (int): Number of subdivisions to compute
        range_start (int): Full range's start index
        range_end (int): Full range's past-the-end index

    Return:
        local_range array[int]: The subdivided interval's index range
        local_offset int: If local range is nonempty, the offset of the
                         subdivided interval's start from the original start, otherwise
                         -1
    """

    assert (
        subdiv_count > 0 and subdiv_id < subdiv_count
    ), f"Invalid subdivision specification id={subdiv_id} count={subdiv_count}"

    assert (
        range_start >= 0 and range_end >= range_start
    ), f"Range specification {range_start}:{range_end} invalid / not providing *absolute* range"

    # Create full index range and split according to subdiv-count
    split_range = np.array_split(np.arange(range_start, range_end), subdiv_count)

    # Grab local range
    local_range = split_range[subdiv_id]

    return local_range, local_range[0] - range_start if len(local_range) else -1


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class HalfClosedInterval:
    """Represents a half closed interval [start, stop) of integers"""

    start: int
    stop: int

    def __post_init__(self) -> None:
        """Makes sure interval is well-defined"""
        assert isinstance(self.start, int)
        assert isinstance(self.stop, int)
        assert self.start <= self.stop

    def __contains__(self, item: int) -> bool:
        """Determines if an item is contained in the interval"""
        return self.start <= item < self.stop

    def __len__(self) -> int:
        """Returns the number of elements in the interval"""
        return self.stop - self.start

    def intersection(self, other: HalfClosedInterval) -> Optional[HalfClosedInterval]:
        """Computes the intersection of two half-closed interval"""
        if other.start >= self.stop or other.stop <= self.start:
            return None

        return HalfClosedInterval(max(self.start, other.start), min(self.stop, other.stop))

    def overlaps(self, other: HalfClosedInterval) -> bool:
        """Checks if the interval has a non-zero overlap with an other closed interval"""
        return self.intersection(other) is not None

    def cover_range(self, sorted_samples: np.ndarray) -> range:
        """Given a set of *sorted* samples (not validated), return the corresponding range for samples
        that are within the interval"""
        if (
            not len(sorted_samples)
            or len(self) == 0
            or not self.intersection(
                # generate closed integer interval [floor(sample[0]), ceil(samples[-1])+1] guaranteed to containing all samples[i]
                HalfClosedInterval(int(np.floor(sorted_samples[0])), int(np.ceil(sorted_samples[-1])) + 1)
            )
        ):
            # empty range for empty samples, empty interval, or missing intersection
            return range(0)

        # non-empty range case
        cover_range_start = np.argmax(self.start <= sorted_samples).item()
        cover_range_stop = (
            np.argmin(sorted_samples < self.stop).item() if self.stop < sorted_samples[-1] else len(sorted_samples)
        )  # full range of frames

        return range(cover_range_start, cover_range_stop)


class Config(object):
    """Simple dictionary holding all options as key/value pairs"""

    def __init__(self, kwargs):
        self.__dict__ = kwargs

    def __iadd__(self, other):
        """Extend with more key/value options"""
        for key, value in other.items():
            self.__dict__[key] = value

        return self
