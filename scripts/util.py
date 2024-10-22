# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import logging

from pathlib import Path
from typing import Final, Optional

import click
import debugpy

import numpy as np
import numpy.typing as npt

from ncore.impl.data.data3 import PointCloudSensor


class NPArrayParamType(click.ParamType):
    name = "NPArray"
    """ Click cmdl argument type for numpy arrays """

    def __init__(self, dim: tuple[int, ...] = (-1,), dtype: npt.DTypeLike = np.float32):
        super().__init__()
        self.dim = dim
        self.dtype = np.dtype(dtype)

    def convert(self, value, param, ctx) -> np.ndarray:
        try:
            return np.fromstring(value.replace("[", "").replace("]", ""), sep=",", dtype=self.dtype).reshape(self.dim)
        except ValueError:
            self.fail(f"{value!r} is not a valid numpy array", param, ctx)


def get_dynamic_flag(sensor: PointCloudSensor, frame_index: int) -> Optional[np.ndarray]:
    """Provides fall-back-based access to the deprecated 'dynamic_flag' point-cloud property"""

    if sensor.has_frame_data(frame_index, "dynamic_flag"):
        # Deprecated 'dynamic_flag' frame property
        return sensor.get_frame_data(frame_index, "dynamic_flag")
    elif sensor.has_frame_generic_data(frame_index, "dynamic_flag"):
        # Generic 'dynamic_flag' frame property property fallback
        return sensor.get_frame_generic_data(frame_index, "dynamic_flag")

    return None


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
