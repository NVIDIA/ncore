# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from pathlib import Path
from typing import TYPE_CHECKING, Final, Optional, Tuple

import click
import debugpy
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


DEBUGPY_EXCEPTION_STR: Final = "Address already in use"


def breakpoint(
    port: int = 5678, log_dir: Optional[Path] = None, allow_port_increment: bool = False, skip_breakpoint: bool = False
) -> None:
    """Open a debugpy port, wait for a remote connection e.g. from VSCode, and break remote debugger

    For help with setup, see the README.md file next to this file.

    Args:
        port: the port on which debugpy will wait for a client to connect.
        log_dir: a directory to which detailed debugpy logs are optionally written.
        allow_port_increment: if True, increment the port when failing on an occupied port.
            this also enables connecting multiple debugger processes to the same target running process.
        skip_breakpoint: don't break remote debugger on this breakpoint after connection
    """

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        debugpy.log_to(log_dir.as_posix())

    if not debugpy.is_client_connected():
        try:
            debugpy.listen(port)
            logging.warning("Waiting for a client to connect to the debugpy port: %d", port)
            debugpy.wait_for_client()
        except RuntimeError as e:
            if any(DEBUGPY_EXCEPTION_STR in arg for arg in e.args) and allow_port_increment:
                logging.info("Debugpy port already in use - incrementing an retrying...")
                port += 1
                breakpoint(port, log_dir)
            else:
                msg = "To find the next free port, use: remote_debug.breakpoint(..., allow_increment=True)"
                logging.info(msg)
                raise e

    if not skip_breakpoint:
        debugpy.breakpoint()
