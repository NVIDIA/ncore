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

from typing import Any, Dict, List, Mapping, Sequence, Union, cast


#: A JSON value as read (e.g., attributes of nodes): of concrete types, dictionaries and lists
Json = Union[Dict[str, "Json"], List["Json"], str, int, float, bool, None]

#: A JSON document as read (a dictionary at the top level)
JsonDocument = Dict[str, Json]

#: A JSON-like value to write (e.g., attributes of nodes): of any mappings / sequences of JSON-like values (e.g.,
#: read-only mappings, tuples), written as JSON objects / arrays (see :func:`to_json`). JSON values are JSON-like
JsonLike = Union[Mapping[str, "JsonLike"], Sequence["JsonLike"], str, int, float, bool, None]

#: A JSON-like document to write (a mapping at the top level, see :data:`JsonLike` and :func:`to_json_document`)
JsonDocumentLike = Mapping[str, JsonLike]


def to_json(value: JsonLike) -> Json:
    """The JSON value of a JSON-like value (recursively): mappings as dictionaries (with string keys), sequences as
    lists (strings, and bytes, which aren't JSON values, are kept: writing bytes fails)"""
    return _to_json(value)


def _to_json(value: Any) -> Json:
    # (untyped: type checkers narrow the recursive JsonLike union differently)
    if isinstance(value, Mapping):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json(item) for item in value]
    return cast(Json, value)


def to_json_document(document: JsonDocumentLike) -> JsonDocument:
    """The JSON document of a mapping of JSON-like values (see :func:`to_json`)"""
    return {str(key): to_json(value) for key, value in document.items()}
