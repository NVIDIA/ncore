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


from pathlib import Path
from typing import Any, cast

import pytest


def test_import_package():
    """Make sure package can be imported"""

    import ncore.data.v4  # noqa: F401


def test_sequence_types():
    """The package exposes the sequence types of the implementation (the same objects)"""
    import ncore.data.v4 as public

    from ncore.impl.data.v4 import components, types

    assert "SequenceMeta" in public.__all__ and public.SequenceMeta is components.SequenceMeta
    assert "ComponentGroupAssignments" in public.__all__
    assert public.ComponentGroupAssignments is types.ComponentGroupAssignments


def test_public_names():
    """The package exposes the V4 APIs of the implementation (the same objects)"""
    import ncore.data.v4 as public

    from ncore.impl.data import itar
    from ncore.impl.data.v4 import compat, components, types

    expected = {
        "SequenceComponentGroupsWriter": components.SequenceComponentGroupsWriter,
        "SequenceComponentGroupsReader": components.SequenceComponentGroupsReader,
        "IndexTailRetention": itar.IndexTailRetention,
        "ComponentWriter": components.ComponentWriter,
        "ComponentReader": components.ComponentReader,
        "BaseSensorComponentWriter": components.BaseSensorComponentWriter,
        "BaseSensorComponentReader": components.BaseSensorComponentReader,
        "BaseRayBundleSensorComponentWriter": components.BaseRayBundleSensorComponentWriter,
        "BaseRayBundleSensorComponentReader": components.BaseRayBundleSensorComponentReader,
        "PosesComponent": components.PosesComponent,
        "IntrinsicsComponent": components.IntrinsicsComponent,
        "MasksComponent": components.MasksComponent,
        "CameraSensorComponent": components.CameraSensorComponent,
        "LidarSensorComponent": components.LidarSensorComponent,
        "RadarSensorComponent": components.RadarSensorComponent,
        "CuboidsComponent": components.CuboidsComponent,
        "PointCloudsComponent": components.PointCloudsComponent,
        "CameraLabelsComponent": components.CameraLabelsComponent,
        "SequenceMeta": components.SequenceMeta,
        "ComponentGroupAssignments": types.ComponentGroupAssignments,
        "SequenceLoaderV4": compat.SequenceLoaderV4,
    }
    assert sorted(public.__all__) == sorted(expected)
    for name, value in expected.items():
        assert getattr(public, name) is value, name


# (positional calls of the sequence reader and loader, untyped as they don't type check)
_PATHS = [Path("missing.zarr.itar")]


def _open_reader():
    import ncore.data.v4

    # (a positional call of the v19 signature: open_consolidated, itar_index_tail_read_size, max_threads)
    cast(Any, ncore.data.v4.SequenceComponentGroupsReader)(_PATHS, True, 1 << 20, 8)


def _open_reader_async():
    import ncore.data.v4

    cast(Any, ncore.data.v4.SequenceComponentGroupsReader).open_async(_PATHS, True)


def _open_loader():
    import ncore.data.v4

    cast(Any, ncore.data.v4.SequenceLoaderV4)(None, "default")


def _open_loader_async():
    import ncore.data.v4

    cast(Any, ncore.data.v4.SequenceLoaderV4).open_async(_PATHS, "default")


@pytest.mark.parametrize("call", [_open_reader, _open_reader_async, _open_loader, _open_loader_async])
def test_keyword_only_parameters(call):
    """The optional parameters of sequence readers and loaders are keyword-only"""
    with pytest.raises(TypeError, match="positional argument"):
        call()


def _keyword_only(function) -> list:
    import inspect

    return [p.name for p in inspect.signature(function).parameters.values() if p.kind is p.KEYWORD_ONLY]


def test_writer_keyword_only_parameters():
    """The optional parameters of the sequence writer and of the point-cloud and camera-label writers are
    keyword-only"""
    import ncore.data.v4 as public

    for function, expected in (
        (public.SequenceComponentGroupsWriter.from_reader, ["store_type", "zarr_format"]),
        (public.SequenceComponentGroupsWriter.__init__, ["store_type", "zarr_format"]),
        (public.PointCloudsComponent.Writer.__init__, ["attribute_schemas"]),
        (public.PointCloudsComponent.Writer.store_pc, ["attributes", "generic_data", "generic_meta_data"]),
        (public.CameraLabelsComponent.Writer.store_label, ["generic_meta_data"]),
    ):
        assert _keyword_only(function) == expected, function.__qualname__
