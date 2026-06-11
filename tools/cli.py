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

from typing import TYPE_CHECKING, Optional, Tuple

import click
import numpy as np


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


class OptionalStrParamType(click.ParamType):
    """Click cmdl argument type for optional strings"""

    name = "OptionalStr"

    def convert(self, value: Optional[str], param, ctx) -> Optional[str]:
        if value is None or value.lower() == "none":
            return None
        return value


class NPArrayParamType(click.ParamType):
    """Click cmdl argument type for numpy arrays"""

    name = "NPArray"

    def __init__(self, dim: Tuple[int, ...] = (-1,), dtype: "npt.DTypeLike" = np.float32):
        super().__init__()
        self.dim = dim
        self.dtype = np.dtype(dtype)

    def convert(self, value: str, param, ctx) -> np.ndarray:
        try:
            return np.fromstring(value.replace("[", "").replace("]", ""), sep=",", dtype=self.dtype).reshape(self.dim)
        except ValueError:
            self.fail(f"{value!r} is not a valid numpy array", param, ctx)


class ResolutionParamType(click.ParamType):
    """Click cmdl argument type for a ``'WxH'`` image resolution -> ``(width, height)``"""

    name = "WxH"

    def convert(self, value: Optional[str], param, ctx) -> Optional[Tuple[int, int]]:
        if value is None:
            return None
        try:
            width_str, height_str = value.lower().split("x")
            width, height = int(width_str), int(height_str)
        except ValueError:
            self.fail(f"{value!r} is not a valid resolution, expected 'WxH' (e.g. '1920x1080')", param, ctx)
        if width <= 0 or height <= 0:
            self.fail(f"resolution must be positive, got {value!r}", param, ctx)
        return (width, height)
