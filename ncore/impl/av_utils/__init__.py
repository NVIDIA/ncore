# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

__version__ = "1.0.0"

import numpy as np

from . import libav_utils_cc  # type: ignore


def isWithin3DBBox(pc: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    # Chech the validity of the input
    assert pc.shape[1] == 3, "Wrong PC input size"
    assert len(bboxes.shape) == 2, "bboxes need to be a 2D numpy array"
    assert bboxes.shape[1] == 9, "bboxes need to be a 2D numpy array"

    return libav_utils_cc._isWithin3DBoundingBox(pc, bboxes)
