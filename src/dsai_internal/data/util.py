# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

## Constants
INDEX_DIGITS = 6  # the number of integer digits to pad counters in output filenames

## Functions
def padded_index_string(index: int, index_digits=INDEX_DIGITS) -> str:
    ''' Pads an integer with leading zeros to a fixed number of digits '''
    return str(index).zfill(index_digits)
