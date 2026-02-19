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

"""Fast-path protobuf parser for Waymo MatrixFloat / MatrixInt32 protos.

This module provides a direct wire-format parser that avoids the slow Python
protobuf repeated-field construction that dominates the cost of ParseFromString
for large matrices (~1M+ elements).
"""

from __future__ import annotations

import zlib

from dataclasses import dataclass

import numpy as np


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode a protobuf varint starting at `pos`, returning (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7


@dataclass
class ParsedMatrix:
    """Lightweight container holding the result of fast-path protobuf matrix parsing."""

    data: np.ndarray
    dims: list[int]


def _decode_zigzag(n: int) -> int:
    """Decode a ZigZag-encoded signed integer (used by protobuf sint32/sint64)."""
    return (n >> 1) ^ -(n & 1)


def _parse_packed_varints_int32(buf: bytes, start: int, end: int) -> list[int]:
    """Decode a sequence of packed varint-encoded int32 values.

    Protobuf encodes negative int32 as 10-byte sign-extended 64-bit varints,
    so we must mask to 32 bits and reinterpret as signed.
    """
    values: list[int] = []
    pos = start
    while pos < end:
        val, pos = _decode_varint(buf, pos)
        val = val & 0xFFFFFFFF
        if val > 0x7FFFFFFF:
            val -= 0x100000000
        values.append(val)
    return values


def parse_matrix_proto(raw_compressed: bytes, dtype: type) -> ParsedMatrix:
    """Fast-path parser for Waymo's MatrixFloat / MatrixInt32 protos.

    Parses the protobuf wire format directly to locate the packed data field
    and extracts it via np.frombuffer (for floats) or varint decoding (for int32),
    avoiding the slow Python protobuf repeated-field construction that dominates
    the cost of ParseFromString for large matrices (~1M+ elements).

    Wire layout (from waymo_open_dataset/dataset.proto):
      MatrixFloat { repeated float data = 1 [packed=true]; optional MatrixShape shape = 2; }
      MatrixInt32 { repeated int32 data = 1 [packed=true]; optional MatrixShape shape = 2; }
      MatrixShape { repeated int32 dims = 1; }

    Note: packed float fields are raw IEEE 754 bytes (4 bytes each), while packed
    int32 fields use varint encoding (variable-length per element).
    """
    buf = zlib.decompress(raw_compressed)
    pos = 0
    end = len(buf)
    data_start: int | None = None
    data_end: int | None = None
    dims: list[int] = []
    is_float = dtype == np.float32 or dtype == np.float64

    while pos < end:
        tag, pos = _decode_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 2:  # length-delimited
            length, pos = _decode_varint(buf, pos)
            if field_number == 1:  # data (packed floats or varints)
                data_start = pos
                data_end = pos + length
            elif field_number == 2:  # shape sub-message
                sub_end = pos + length
                while pos < sub_end:
                    sub_tag, pos = _decode_varint(buf, pos)
                    sub_field = sub_tag >> 3
                    sub_wire = sub_tag & 0x07
                    if sub_field == 1 and sub_wire == 0:
                        val, pos = _decode_varint(buf, pos)
                        dims.append(val)
                    elif sub_field == 1 and sub_wire == 2:
                        sub_len, pos = _decode_varint(buf, pos)
                        sub_data_end = pos + sub_len
                        while pos < sub_data_end:
                            val, pos = _decode_varint(buf, pos)
                            dims.append(val)
                    else:
                        if sub_wire == 0:
                            _, pos = _decode_varint(buf, pos)
                        elif sub_wire == 2:
                            sl, pos = _decode_varint(buf, pos)
                            pos += sl
                        else:
                            raise ValueError(f"Unexpected sub wire type {sub_wire}")
                continue  # pos already advanced past sub-message
            pos += length
        elif wire_type == 0:  # varint
            _, pos = _decode_varint(buf, pos)
        elif wire_type == 5:  # 32-bit
            pos += 4
        elif wire_type == 1:  # 64-bit
            pos += 8
        else:
            raise ValueError(f"Unexpected wire type {wire_type}")

    if data_start is None or data_end is None:
        return ParsedMatrix(data=np.empty(0, dtype=dtype), dims=dims)

    data: np.ndarray
    if is_float:
        data = np.frombuffer(memoryview(buf)[data_start:data_end], dtype=dtype)
    else:
        data = np.array(_parse_packed_varints_int32(buf, data_start, data_end), dtype=dtype)

    return ParsedMatrix(data=data, dims=dims)
