# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import sys


# There are potential conflicts between protobuf versions incorporated as an (outdated)
# tensorflow dependency, and the actual ones we'd like to use directly from the proto rules - remove
# the internal ones (which have a <pip-hub>_<pyversion>_protobuf_... path component)
sys_path = sys.path  # push current paths
sys.path = [p for p in sys.path if "_protobuf_" not in p]

import tensorflow.compat.v1 as tf

from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2


sys.path = sys_path  # pop modified paths back to original
