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

import unittest


class TestPackage(unittest.TestCase):
    def test_public_names(self):
        """The package exposes the geometry utilities of the implementation (the same objects)"""
        import ncore.geometry as public

        from ncore.impl.common import transformations

        expected = [
            "PoseInterpolator",
            "PoseGraphInterpolator",
            "MotionCompensator",
            "MotionCompensationMethod",
            "MotionCompensatorParameters",
            "se3_inverse",
            "so3_trans_2_se3",
            "transform_point_cloud",
            "bbox_pose",
            "pose_bbox",
            "transform_bbox",
            "is_within_3d_bboxes",
            "HalfClosedInterval",
            "time_bounds",
        ]
        self.assertEqual(sorted(public.__all__), sorted(expected))
        for name in expected:
            self.assertIs(getattr(public, name), getattr(transformations, name), name)
