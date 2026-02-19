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

"""
Proxy module for Waymo Open Dataset dependencies.

There are potential conflicts between protobuf versions incorporated as an (outdated)
tensorflow dependency, and the actual ones we'd like to use directly from the proto rules.
This module removes the internal ones (which have a <pip-hub>_<pyversion>_protobuf_... path
component) to avoid conflicts.
"""

import sys


# Push current paths and filter out conflicting protobuf paths
sys_path = sys.path
sys.path = [p for p in sys.path if "_protobuf_" not in p]

import tensorflow.compat.v1 as tf

from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2
from waymo_open_dataset.utils import range_image_utils as waymo_range_image_utils
from waymo_open_dataset.utils import transform_utils as waymo_transform_utils


# Pop modified paths back to original
sys.path = sys_path
