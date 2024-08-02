# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

from typing import Optional

import click

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
