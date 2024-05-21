# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

__version__ = "1.0.0"

import sys

import numpy as np

if sys.version_info >= (3, 9):
    from . import libav_utils_cc as av_utils_cc  # type: ignore
else:
    from . import libav_utils_cc_3_8 as av_utils_cc  # type: ignore


def isWithin3DBBoxes(pc: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    # Check the validity of the input
    assert pc.shape[1] == 3, "Wrong PC input size"
    assert len(bboxes.shape) == 2, "bboxes need to be a 2D numpy array"
    assert bboxes.shape[1] == 9, "bboxes need to be a 2D numpy array"

    return av_utils_cc._isWithin3DBBoxes(pc, bboxes)
