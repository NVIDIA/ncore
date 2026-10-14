# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""asyncio helpers (independent of the python version)"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import sys

from typing import Callable, TypeVar


if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:  # pragma: no cover
    from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


async def run_in_thread(func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Runs a blocking function on a worker thread of the running event loop's default executor, without blocking the
    loop (``asyncio.to_thread`` of python >= 3.9, which also propagates context variables to the thread)"""
    if sys.version_info >= (3, 9):
        return await asyncio.to_thread(func, *args, **kwargs)
    context = contextvars.copy_context()  # pragma: no cover
    call = functools.partial(context.run, func, *args, **kwargs)  # pragma: no cover
    return await asyncio.get_running_loop().run_in_executor(None, call)  # pragma: no cover
