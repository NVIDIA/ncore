# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import unittest

from .util import closest_index_sorted


class TestClosestIndexSorted(unittest.TestCase):
    """Test to verify functionality of closest_index_sorted"""

    def test_empty(self):
        with self.assertRaises(ValueError):
            closest_index_sorted([], 5)  # empty array -> raises exception

    def test_regular(self):
        def check(sorted_array, value, expected_index: int):
            assert closest_index_sorted(sorted_array, value) == expected_index

        sorted_timestamp_array = [
            1624564702900262,
            1624564703000172,
            1624564703100110,
            1624564703200048,
            1624564703299986,
            1624564703399952,
        ]

        check(sorted_timestamp_array, sorted_timestamp_array[0], 0)  # exact first
        check(sorted_timestamp_array, sorted_timestamp_array[0] - 1, 0)  # slightly smaller than first
        check(sorted_timestamp_array, sorted_timestamp_array[0] + 1, 0)  # slightly larger than first

        check(sorted_timestamp_array, sorted_timestamp_array[-1], len(sorted_timestamp_array) - 1)  # exact last
        check(
            sorted_timestamp_array, sorted_timestamp_array[-1] - 1, len(sorted_timestamp_array) - 1
        )  # slightly smaller than last
        check(
            sorted_timestamp_array, sorted_timestamp_array[-1] + 1, len(sorted_timestamp_array) - 1
        )  # slightly larger than last

        for idx in range(len(sorted_timestamp_array)):
            check(sorted_timestamp_array, sorted_timestamp_array[idx], idx)  # exact hit
            check(sorted_timestamp_array, sorted_timestamp_array[idx] - 1, idx)  # inexact hit
            check(sorted_timestamp_array, sorted_timestamp_array[idx] + 1, idx)  # inexact hit
