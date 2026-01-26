# Copyright 2019 The Waymo Open Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ========================================================================
#
# Modifications Copyright 2026 NVIDIA CORPORATION

from typing import Optional

import numpy as np

from scipy.linalg import expm

from scripts.data_converter.waymo.deps import dataset_pb2, tf


## Functions are adapted from the official waymo open github page
def extrapolate_pose_based_on_velocity(
    T_SDC_global: np.ndarray, v_global: np.ndarray, w_global: np.ndarray, dt_sec: float
) -> np.ndarray:
    T_extrapolated = np.eye(4, dtype=np.float32)
    T_extrapolated[:3, :3] = expm(get_skew_symmetric(w_global) * dt_sec) @ T_SDC_global[:3, :3]
    T_extrapolated[:3, 3:4] = T_SDC_global[:3, 3:4] + v_global * dt_sec

    return T_extrapolated


def get_skew_symmetric(vec: np.ndarray) -> np.ndarray:
    skew_sym = np.zeros((3, 3), dtype=np.float32)

    skew_sym[0, 1] = -vec[2]
    skew_sym[0, 2] = vec[1]
    skew_sym[1, 0] = vec[2]
    skew_sym[1, 2] = -vec[0]
    skew_sym[2, 0] = -vec[1]
    skew_sym[2, 1] = vec[0]

    return skew_sym


### From here on the function are adopted from the waymo utils
def parse_range_image_and_segmentations(frame, laser_name, ri_index: int = 0):
    """Parse range images and segmentations given a frame.

    Args:
       frame: open dataset frame proto
       laser_name: the name of the laser sensor to process
       ri_index: 0 for the first return, 1 for the second return.

    Returns:
       range_image: range_image_for_return.
       segmentations: Optional segmentations_for_return.
       range_image_top_pose: range image pixel pose for top lidar
    """

    range_image = dataset_pb2.MatrixFloat()
    segmentation: Optional[dataset_pb2.MatrixInt32] = None
    range_image_top_pose = dataset_pb2.MatrixFloat()

    for laser in frame.lasers:
        laser_ri_return = laser.ri_return1 if ri_index == 0 else laser.ri_return2

        if laser.name == dataset_pb2.LaserName.TOP:
            range_image_top_pose_str_tensor = tf.io.decode_compressed(
                laser_ri_return.range_image_pose_compressed, "ZLIB"
            )
            range_image_top_pose.ParseFromString(bytearray(range_image_top_pose_str_tensor.numpy()))

        if laser.name != laser_name:
            continue

        if len(laser_ri_return.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(laser_ri_return.range_image_compressed, "ZLIB")
            range_image.ParseFromString(bytearray(range_image_str_tensor.numpy()))

        if len(laser_ri_return.segmentation_label_compressed) > 0:
            segmentations_str_tensor = tf.io.decode_compressed(laser_ri_return.segmentation_label_compressed, "ZLIB")
            segmentation = dataset_pb2.MatrixInt32()
            segmentation.ParseFromString(bytearray(segmentations_str_tensor.numpy()))

    return range_image, segmentation, range_image_top_pose


def convert_range_image_to_point_cloud(
    frame,
    laser_name: str,
    range_image,
    segmentation,
    range_image_top_pose,
    start_end_timestamps,
):
    """Convert range images to point cloud.

    Args:
      frame: open dataset frame.
      laser_name: the name of the laser sensor to process.
      range_image: range_image return.
      segmentation: segmentation return.
      range_image_top_pose: range image pixel pose for top lidar.
      start_end_timestamps: timestamp bounds to approximate point timestamps from.

    Returns:
      points: [N, 8] list of 3d lidar points of length 8, floats (signals: start-ray-point, end-ray-point, intensity, elongation).
      segs: [N, 2] list of point instance-IDs / semantic_classes (if available)
      timestamps: [N] list of point timestamps, unsigned-integers
      range_image_indices: [N, 2] per-point indices in the original range-image
      inclinations: [H] per-row ray inclinations [rad]
      azimuths: [W] per-column ray azimuths [rad]

    """
    cartesian_range_image, timestamps_full, inclinations, azimuths = convert_range_image_to_cartesian(
        frame, laser_name, range_image, range_image_top_pose, start_end_timestamps
    )

    points: np.ndarray = np.empty(0)
    segs: Optional[np.ndarray] = None
    timestamps: np.ndarray = np.empty(0)
    range_image_indices: np.ndarray = np.empty(0)

    for c in frame.context.laser_calibrations:
        if c.name != laser_name:
            continue

        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0  # filter points based on positive range

        range_image_indices_tensor = tf.compat.v1.where(range_image_mask)

        points_tensor = tf.gather_nd(cartesian_range_image, range_image_indices_tensor)
        points = points_tensor.numpy()

        if segmentation:
            seq_tensor = tf.reshape(tf.convert_to_tensor(value=segmentation.data), segmentation.shape.dims)
            seg_points_tensor = tf.gather_nd(seq_tensor, range_image_indices_tensor)
            segs = seg_points_tensor.numpy()

        timestamps_tensor = tf.gather_nd(timestamps_full, range_image_indices_tensor)
        timestamps = timestamps_tensor.numpy()

        range_image_indices = range_image_indices_tensor.numpy().astype(np.uint32)

    return (
        points,
        segs,
        timestamps,
        range_image_indices,
        inclinations.numpy().astype(np.float32),
        azimuths.numpy().astype(np.float32),
    )


def convert_range_image_to_cartesian(frame, laser_name, range_image, range_image_top_pose, start_end_timestamps):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Args:
      frame: open dataset frame.
      laser_name: the name of the laser sensor to process.
      range_image: range_image_return.
      range_image_top_pose: range image pixel pose for top lidar.
      start_end_timestamps: timestamp bounds to approximate point timestamps from.

    Returns:
      (H, W, 8) range images in Cartesian coordinates. (signals: start-ray-point, end-ray-point, intensity, elongation).
    """
    cartesian_range_images = None
    timestamps = None

    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims,
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    for c in frame.context.laser_calibrations:
        if c.name != laser_name:
            continue

        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)

        range_image_cartesian, range_image_timestamps, azimuths = extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            start_end_timestamps,
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        intensity = range_image_tensor[..., 1]
        elongation = range_image_tensor[..., 2]

        end_point_cartesian = tf.squeeze(range_image_cartesian[0], axis=0)
        source_point_cartesian = tf.squeeze(range_image_cartesian[1], axis=0)

        cartesian_range_images = tf.concat(
            [
                source_point_cartesian,
                end_point_cartesian,
                intensity[:, :, None],
                elongation[:, :, None],
            ],
            axis=-1,
        )

        timestamps = tf.squeeze(range_image_timestamps, axis=0)
        azimuths = tf.squeeze(azimuths, axis=0)

    return cartesian_range_images, timestamps, beam_inclinations, azimuths


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    start_end_timestamps,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      start_end_timestamps [2]: timestamp bounds to approximate point timestamps from.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in world frame.
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar, azimuths = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
        range_image_cartesian, timestamps = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            start_end_timestamps,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
            return_local_coordinates=False,
        )

        return range_image_cartesian, timestamps, azimuths


def compute_inclination(inclination_range, height, scope=None):
    """Computes uniform inclination range based the given range and height.

    Args:
      inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
        inclination].
      height: an integer indicates height of the range image.
      scope: the name scope.

    Returns:
      inclination: [..., height] tensor. Inclinations computed.
    """
    with tf.compat.v1.name_scope(scope, "ComputeInclination", [inclination_range]):
        diff = inclination_range[..., 1] - inclination_range[..., 0]
        inclination = (0.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) / tf.cast(
            height, dtype=inclination_range.dtype
        ) * tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1]
        return inclination


def compute_range_image_polar(range_image, extrinsic, inclination, dtype=tf.float32, scope=None):
    """Computes range image polar coordinates.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_polar: [B, H, W, 3] polar coordinates.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    _, height, width = _combined_static_and_dynamic_shape(range_image)
    range_image_dtype = range_image.dtype
    range_image = tf.cast(range_image, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    inclination = tf.cast(inclination, dtype=dtype)

    with tf.compat.v1.name_scope(scope, "ComputeRangeImagePolar", [range_image, extrinsic, inclination]):
        with tf.compat.v1.name_scope("Azimuth"):
            # [B].
            az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
            # [W].
            ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - 0.5) / tf.cast(width, dtype=dtype)
            # [B, W].
            azimuth = (ratios * 2.0 - 1.0) * np.pi - tf.expand_dims(az_correction, -1)

        # [B, H, W]
        azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
        # [B, H, W]
        inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
        range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image], axis=-1)
        return tf.cast(range_image_polar, dtype=range_image_dtype), azimuth


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    start_end_timestamps,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
    return_local_coordinates=False,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        ## Approximate point timestamps (see https://github.com/waymo-research/waymo-open-dataset/issues/619#issuecomment-1496410092)
        #  inferred from column index [B, H, W]
        nRows, nCols = azimuth.shape[1:]
        column_relative_times = tf.reshape(tf.range(nCols) / (nCols - 1), (-1, 1, nCols))
        relative_times = tf.repeat(column_relative_times, nRows, axis=1)
        # *clockwise* spinning lidar lidar (largest azimuth are measured first)
        range_image_timestamps = start_end_timestamps[0] + tf.cast(
            relative_times * (start_end_timestamps[1] - start_end_timestamps[0]),
            dtype=start_end_timestamps[0].dtype,
        )

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_center = tf.zeros_like(range_image_points)

        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation

        range_image_center = tf.einsum("bkr,bijr->bijk", rotation, range_image_center) + translation
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points) + pixel_pose_translation
            )

            range_image_center = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_center) + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")

            if return_local_coordinates:
                # To vehicle frame corresponding to the given frame_pose
                # [B, 4, 4]
                world_to_vehicle = tf.linalg.inv(frame_pose)
                world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
                world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
                # [B, H, W, 3]
                range_image_points = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )
                range_image_center = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_center)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_center = tf.cast(range_image_center, dtype=range_image_polar_dtype)

        return (range_image_points, range_image_center), range_image_timestamps


def get_rotation_matrix(roll, pitch, yaw, name=None):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Args:
      roll : x-rotation in radians.
      pitch: y-rotation in radians. The shape must be the same as roll.
      yaw: z-rotation in radians. The shape must be the same as roll.
      name: the op name.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """
    with tf.compat.v1.name_scope(name, "GetRotationMatrix", [yaw, pitch, roll]):
        cos_roll = tf.cos(roll)
        sin_roll = tf.sin(roll)
        cos_yaw = tf.cos(yaw)
        sin_yaw = tf.sin(yaw)
        cos_pitch = tf.cos(pitch)
        sin_pitch = tf.sin(pitch)

        ones = tf.ones_like(yaw)
        zeros = tf.zeros_like(yaw)

        r_roll = tf.stack(
            [
                tf.stack([ones, zeros, zeros], axis=-1),
                tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
                tf.stack([zeros, sin_roll, cos_roll], axis=-1),
            ],
            axis=-2,
        )
        r_pitch = tf.stack(
            [
                tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
                tf.stack([zeros, ones, zeros], axis=-1),
                tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
            ],
            axis=-2,
        )
        r_yaw = tf.stack(
            [
                tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
                tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
                tf.stack([zeros, zeros, ones], axis=-1),
            ],
            axis=-2,
        )

        return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

    Args:
      rotation: [..., N, N] rotation tensor.
      translation: [..., N] translation tensor. This must have the same type as
        rotation.

    Returns:
      transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
        rotation.
    """
    with tf.name_scope("GetTransform"):
        # [..., N, 1]
        translation_n_1 = translation[..., tf.newaxis]
        # [..., N, N+1]
        transform = tf.concat([rotation, translation_n_1], axis=-1)
        # [..., N]
        last_row = tf.zeros_like(translation)
        # [..., N+1]
        last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
        # [..., N+1, N+1]
        transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
        return transform


def _combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape
