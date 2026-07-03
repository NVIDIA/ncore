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

"""Proxy module for Waymo Open Dataset dependencies."""

import tensorflow.compat.v1 as tf  # ty:ignore[unresolved-import]

from waymo_open_dataset import dataset_pb2, label_pb2  # ty:ignore[unresolved-import]
from waymo_open_dataset.protos import camera_segmentation_pb2  # ty:ignore[unresolved-import]
from waymo_open_dataset.utils import range_image_utils as waymo_range_image_utils  # ty:ignore[unresolved-import]
from waymo_open_dataset.utils import transform_utils as waymo_transform_utils  # ty:ignore[unresolved-import]


# This module deliberately re-exports the Waymo Open Dataset symbols above so
# that the rest of the converter imports them from a single proxy location.
__all__ = [
    "tf",
    "dataset_pb2",
    "label_pb2",
    "camera_segmentation_pb2",
    "waymo_range_image_utils",
    "waymo_transform_utils",
]
