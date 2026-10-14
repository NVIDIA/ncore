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

"""Compressed binary encodings of JSON documents (record indices of indexed tar archives, compressed consolidated zarr
metadata)"""

from __future__ import annotations

import io
import lzma
import zlib

from typing import cast

import cbor2

from ncore.impl.data.json_types import JsonDocument


def encode_cbor_xz(document: JsonDocument) -> bytes:
    """Encodes a document as CBOR + LZMA (xz container format)"""
    with io.BytesIO() as buffer:
        with lzma.open(buffer, "wb", format=lzma.FORMAT_XZ) as lzma_file:
            cbor2.dump(document, lzma_file)
        return buffer.getvalue()


def decode_cbor_xz(data: bytes) -> JsonDocument:
    """Decodes a CBOR + LZMA (xz container format) encoded document"""
    return _document(cbor2.loads(lzma.LZMADecompressor().decompress(data)))


def encode_cbor_zlib(document: JsonDocument) -> bytes:
    """Encodes a document as CBOR + zlib (decodes several times faster than CBOR + LZMA)"""
    return zlib.compress(cbor2.dumps(document), 9)


def decode_cbor_zlib(data: bytes) -> JsonDocument:
    """Decodes a CBOR + zlib encoded document"""
    return _document(cbor2.loads(zlib.decompress(data)))


def _document(value: object) -> JsonDocument:
    if not isinstance(value, dict):
        raise ValueError(f"Not an encoded JSON document: {type(value).__name__}")
    return cast(JsonDocument, value)
