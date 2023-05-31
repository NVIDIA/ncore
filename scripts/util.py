# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click

import numpy as np
import numpy.typing as npt

class NPArrayParamType(click.ParamType):
    ''' Click cmdl argument type for numpy arrays '''
    def __init__(self, dim: tuple[int, ...] = (-1,), dtype: npt.DTypeLike = np.float32):
        super().__init__()
        self.dim = dim
        self.dtype = np.dtype(dtype)

    def convert(self, value, param, ctx) -> np.ndarray:
        try:
            return np.fromstring(value.replace('[','').replace(']',''), sep=',', dtype=self.dtype).reshape(self.dim)
        except ValueError:
            self.fail(f"{value!r} is not a valid numpy array", param, ctx)
