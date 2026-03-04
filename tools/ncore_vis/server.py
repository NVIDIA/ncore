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

"""Viser server lifecycle and per-client renderer management."""

from __future__ import annotations

import logging

from typing import Dict, Optional

import viser

from ncore.impl.data.compat import SequenceLoaderProtocol
from tools.ncore_vis.data_loader import DataLoader
from tools.ncore_vis.renderer import NCoreVisRenderer


logger = logging.getLogger(__name__)


class NCoreVisServer:
    """Manages the viser server, the data loader, and per-client renderers.

    On each client connection, a new :class:`NCoreVisRenderer` is created with
    independent GUI and scene state.
    """

    def __init__(
        self,
        loader: SequenceLoaderProtocol,
        host: str,
        port: int,
        rig_frame_id: Optional[str] = "rig",
        world_frame_id: str = "world",
    ) -> None:
        self._loader: SequenceLoaderProtocol = loader
        self._host: str = host
        self._port: int = port
        self._rig_frame_id: Optional[str] = rig_frame_id
        self._world_frame_id: str = world_frame_id

    def start(self) -> None:
        """Create the data loader, start the viser server, and block forever."""
        self._data_loader = DataLoader(
            self._loader, rig_frame_id=self._rig_frame_id, world_frame_id=self._world_frame_id
        )
        self._renderers: Dict[int, NCoreVisRenderer] = {}

        server = viser.ViserServer(host=self._host, port=self._port)
        server.on_client_connect(self._on_connect)
        server.on_client_disconnect(self._on_disconnect)

        logger.info("NCore Viewer started on %s:%d", self._host, self._port)

        server.sleep_forever()

    def _on_connect(self, client: viser.ClientHandle) -> None:
        logger.info("Client %d connected", client.client_id)
        self._renderers[client.client_id] = NCoreVisRenderer(
            client=client,
            data_loader=self._data_loader,
        )

    def _on_disconnect(self, client: viser.ClientHandle) -> None:
        logger.info("Client %d disconnected", client.client_id)
        self._renderers.pop(client.client_id, None)
