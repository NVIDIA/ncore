# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import tempfile
from src.py.common.common import load_pkl, save_pkl


def test_save_load_pkl():
    """ Test to verify functionality of load_pkl / save_pkl """
    def check(obj):
        with tempfile.NamedTemporaryFile() as tmp:
            save_pkl(obj, tmp.name)
            assert obj == load_pkl(
                tmp.name
            ), "serialized object not equal to de-serialized version"

    check({})
    check({'some': 'entry', 'some-int': 5})
    check({'something': [1, 2, [3, 4]], 'some-int': {'foo': 'bar'}})
