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

"""pai-clip-dl: Download and stream clip data from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset."""

from pai_clip_dl.config import Config
from pai_clip_dl.downloader import ClipDownloader
from pai_clip_dl.index import ClipIndex, FeatureSpec
from pai_clip_dl.remote import HFRemote
from pai_clip_dl.streaming import StreamingZipAccess


__all__ = [
    "Config",
    "HFRemote",
    "ClipIndex",
    "FeatureSpec",
    "StreamingZipAccess",
    "ClipDownloader",
]
