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

"""JSON-like values (independent of the other data modules)"""

from __future__ import annotations

from typing import Dict, List, Union


#: A JSON-serializable value
JsonLike = Union[
    Dict[str, "JsonLike"],
    List["JsonLike"],
    str,
    int,
    float,
    bool,
    None,
]

#: A JSON-serializable document (a dictionary at the top level)
JsonDocument = Dict[str, JsonLike]
