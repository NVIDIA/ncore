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

import os
import re
import sys

from sphinx.cmd.build import main


if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    # Sphinx evaluates conf.py from the configuration directory: resolve the execroot-relative path of the build
    # information file (see defs.bzl) before it changes directories
    if "SPHINX_BUILD_INFO_FILE" in os.environ:
        os.environ["SPHINX_BUILD_INFO_FILE"] = os.path.abspath(os.environ["SPHINX_BUILD_INFO_FILE"])
    sys.exit(main())
