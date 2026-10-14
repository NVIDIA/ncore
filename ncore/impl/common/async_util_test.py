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

import asyncio
import contextvars
import threading
import unittest

from ncore.impl.common.async_util import run_in_thread


VARIABLE: contextvars.ContextVar[str] = contextvars.ContextVar("variable", default="unset")


class TestRunInThread(unittest.TestCase):
    def test_runs_on_a_worker_thread(self):
        """Functions run on a worker thread (with arguments, keyword arguments and context variables), the result and
        exceptions are returned to the awaiting coroutine"""

        def work(a: int, b: int = 0) -> tuple:
            return a + b, threading.current_thread().name, VARIABLE.get()

        async def main() -> tuple:
            VARIABLE.set("set")
            return await run_in_thread(work, 1, b=2)

        result, thread, variable = asyncio.run(main())
        self.assertEqual((result, variable), (3, "set"))
        self.assertNotEqual(thread, threading.current_thread().name)

        def fail() -> None:
            raise KeyError("missing")

        with self.assertRaises(KeyError):
            asyncio.run(run_in_thread(fail))
