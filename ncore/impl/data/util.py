# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from dataclasses import field
from typing import TYPE_CHECKING

import dataclasses_json
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

## Constants
INDEX_DIGITS = 6  # the number of integer digits to pad counters in output filenames


## Functions
def padded_index_string(index: int, index_digits=INDEX_DIGITS) -> str:
    ''' Pads an integer with leading zeros to a fixed number of digits '''
    return str(index).zfill(index_digits)


def closest_index_sorted(sorted_array: np.ndarray, value: int) -> int:
    ''' Returns the index of the closest value within a *sorted* array relative to a query value.
    
        Note: we are *not* checking that the input is sorted
    '''
    if not len(sorted_array):
        raise ValueError('input array is empty')

    idx = int(np.searchsorted(sorted_array, value, side="left"))

    if idx > 0:
        if idx == len(sorted_array):
            return idx - 1
        if abs(value - sorted_array[idx - 1]) < abs(sorted_array[idx] - value):
            return idx - 1

    return idx


def numpy_array_field(datatype: 'npt.DTypeLike', default=None):
    ''' Provides encoder / decoder functionality for numpy arrays into field types compatible with dataclass-JSON '''
    def decoder(*args, **kwargs):
        return np.array(*args, dtype=datatype, **kwargs)

    return field(default=default, metadata=dataclasses_json.config(encoder=np.ndarray.tolist, decoder=decoder))


def enum_field(enum_class, default=None):
    ''' Provides encoder / decoder functionality for enum types into field types compatible with dataclass-JSON '''
    def encoder(variant):
        ''' encode enum as name's string representation. This way values in JSON are "human-readable '''
        return variant.name

    def decoder(variant):
        ''' load enum variant from name's string to value map of the enumeration type '''
        return enum_class.__members__[variant]

    return field(default=default, metadata=dataclasses_json.config(encoder=encoder, decoder=decoder))
