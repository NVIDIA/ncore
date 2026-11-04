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
