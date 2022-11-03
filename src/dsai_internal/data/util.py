# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import math

from typing import Any

import numpy as np

## Constants
INDEX_DIGITS = 6  # the number of integer digits to pad counters in output filenames

## Functions
def padded_index_string(index: int, index_digits=INDEX_DIGITS) -> str:
    ''' Pads an integer with leading zeros to a fixed number of digits '''
    return str(index).zfill(index_digits)


def closest_index_sorted(sorted_array: np.ndarray, value: Any) -> int:
    ''' Returns the index of the closest value within a *sorted* array relative to a query value.
    
        Note: we are not checking that the input is sorted
    '''
    if not sorted_array:
        raise ValueError('input array is empty')

    idx = int(np.searchsorted(sorted_array, value, side="left"))

    if idx > 0 and (idx == len(sorted_array)
                    or math.fabs(value - sorted_array[idx - 1]) < math.fabs(value - sorted_array[idx])):
        return idx - 1
    else:
        return idx
