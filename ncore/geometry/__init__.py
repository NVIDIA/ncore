# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package exposing NCore's geometry utilities: pose interpolation and motion compensation, rigid transformations
of points and bounding boxes, and time intervals"""

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    MotionCompensationMethod,
    MotionCompensator,
    MotionCompensatorParameters,
    PoseGraphInterpolator,
    PoseInterpolator,
    bbox_pose,
    is_within_3d_bboxes,
    pose_bbox,
    se3_inverse,
    so3_trans_2_se3,
    time_bounds,
    transform_bbox,
    transform_point_cloud,
)


__all__ = [
    # poses
    "PoseInterpolator",
    "PoseGraphInterpolator",
    "MotionCompensator",
    "MotionCompensationMethod",
    "MotionCompensatorParameters",
    # rigid transformations
    "se3_inverse",
    "so3_trans_2_se3",
    "transform_point_cloud",
    # bounding boxes
    "bbox_pose",
    "pose_bbox",
    "transform_bbox",
    "is_within_3d_bboxes",
    # time intervals
    "HalfClosedInterval",
    "time_bounds",
]
