# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import tempfile
import unittest

from pathlib import Path

import numpy as np

from parameterized import parameterized
from python.runfiles import Runfiles

from .conversion import Data3Converter

from ncore.impl.data.data3 import ShardDataLoader


_RUNFILES = Runfiles.Create()


class TestData3Converter(unittest.TestCase):
    """Test to verify functionality of V3->V4 data converter"""

    def setUp(self):
        # from scripts.util import breakpoint

        # breakpoint()

        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # load V3 reference data
        all_shards = sorted(
            [
                str(p)
                for p in Path(
                    _RUNFILES.Rlocation(
                        "test-data-v3-shards/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_0-3.zarr.itar"
                    )
                ).parent.iterdir()
                if p.match("*.itar")
            ]
        )

        self.assertEqual(len(all_shards), 3)

        self.shard_data_loader = ShardDataLoader(all_shards)

    @parameterized.expand(
        [
            ("itar",),
            ("itar",),
            ("directory",),
            ("directory",),
        ]
    )
    def test_convert(
        self,
        store_type,
    ):
        """Test to make sure serialized data is faithfully reloaded"""

        tempdir = tempfile.TemporaryDirectory()

        ## Convert reference sequence
        Data3Converter.convert(
            self.shard_data_loader,
            target_output_dir_path=Path(tempdir.name),
            store_type=store_type,
        )
