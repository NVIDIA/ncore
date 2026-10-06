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

from __future__ import annotations

import ast
import concurrent.futures
import io
import json
import logging
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import dataclasses_json
import numpy as np
import PIL.Image as PILImage

from typing_extensions import Concatenate, ParamSpec
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.common.util import MD5Hasher, unpack_optional
from ncore.impl.data import nodes, stores, types, util
from ncore.impl.data.types import PointCloud


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]

if sys.version_info >= (3, 11):
    # Older python versions have issues with type-hints for nested types in
    # combination with typing.get_type_hints() (used by, e.g., 'dataclasses_json')
    # - alias these globally as a workaround
    from typing import Self

_logger = logging.getLogger(__name__)

VERSION = "v4"

# Compression of sensor frame data, which is fast to decode on modern hardware (blosc lz4, level 5, bit-shuffle)
_FAST_DECODE_COMPRESSION = nodes.BloscCompression(cname="lz4", clevel=5, shuffle="bitshuffle")


def _normalize_chunks(
    shape: "Tuple[int, ...]",
    *,
    require_nonzero_dims: "Optional[Tuple[int, ...]]" = None,
) -> "Tuple[int, ...]":
    """Normalize a chunk shape by replacing zero dimensions with 1.

    Zarr v2 rejects chunk tuples containing zeros (division by zero in internal
    bookkeeping). This helper ensures that every dimension in the chunk tuple is
    at least 1 while the *logical* data shape passed to ``create_array`` via
    the ``data=`` argument remains unchanged.

    Args:
        shape: The data shape to derive a chunk shape from.
        require_nonzero_dims: Optional tuple of dimension indices that MUST be
            non-zero (raises AssertionError otherwise). When ``None`` (the
            default) all dimensions are allowed to be zero and get clamped to 1
            in the returned chunk shape.

    Returns:
        A chunk-safe tuple where any zero dimension is replaced with 1.
    """
    if require_nonzero_dims is not None:
        for d in require_nonzero_dims:
            assert shape[d] != 0, f"Dimension {d} must be non-zero, got shape {shape}"
    return tuple(max(1, s) for s in shape)


@dataclass
class SequenceMeta(dataclasses_json.DataClassJsonMixin):
    """Strongly typed representation for V4 sequence metadata"""

    @dataclass
    class TimestampInterval(dataclasses_json.DataClassJsonMixin):
        """Timestamp interval with start and stop times in microseconds"""

        start: int
        stop: int

    @dataclass
    class ComponentInstanceMeta(dataclasses_json.DataClassJsonMixin):
        """Component instance metadata"""

        version: str
        generic_meta_data: Dict[str, Any]

    @dataclass
    class ComponentStoreMeta(dataclasses_json.DataClassJsonMixin):
        """Component store metadata"""

        path: str
        md5: str
        components: Dict[str, Dict[str, "SequenceMeta.ComponentInstanceMeta"]]

    sequence_id: str
    sequence_timestamp_interval_us: SequenceMeta.TimestampInterval
    generic_meta_data: Dict[str, Any]
    version: str
    component_stores: List[SequenceMeta.ComponentStoreMeta]


class SequenceComponentGroupsWriter:
    """SequenceComponentGroupsWriter manages store groups for writing for NCore V4 / zarr data components for a single NCore sequence"""

    @staticmethod
    def from_reader(
        output_dir_path: UPath,
        store_base_name: str,
        sequence_reader: SequenceComponentGroupsReader,
        store_type: Literal["itar", "directory"] = "itar",  # valid values: ['itar', 'directory']
        zarr_format: Literal[2, 3] = 2,
    ) -> SequenceComponentGroupsWriter:
        """Creates a SequenceComponentGroupsWriter from an existing SequenceComponentGroupsReader instance to share consistent per-sequence meta-data"""

        return SequenceComponentGroupsWriter(
            output_dir_path=output_dir_path,
            store_base_name=store_base_name,
            sequence_id=sequence_reader.sequence_id,
            sequence_timestamp_interval_us=sequence_reader.sequence_timestamp_interval_us,
            generic_meta_data=sequence_reader.generic_meta_data,
            store_type=store_type,
            zarr_format=zarr_format,
        )

    def __init__(
        self,
        output_dir_path: UPath,
        store_base_name: str,
        # Identifier of the sequence
        sequence_id: str,
        # The time range for the sequence
        sequence_timestamp_interval_us: HalfClosedInterval,
        # Generic sequence meta-data (needs to be json-serializable) - will be stored into each component store group
        generic_meta_data: Dict[str, types.JsonLike],
        # Zarr store type: either serialize as .itar archive store (default / production) or plain "directory" store (simpler for introspection / asynchronous / external setup)
        store_type: Literal["itar", "directory"] = "itar",  # valid values: ['itar', 'directory']
        # On-disk zarr format of the component stores: format v2 (default) is readable by all ncore versions,
        # format v3 requires zarr-python>=3 for writing *and* reading
        zarr_format: Literal[2, 3] = 2,
    ):
        """
        Instantiate sequence component groups writer and initialize the default data groups and file stores for a given sequence and sensor IDs
        """

        self._output_dir_path = output_dir_path
        self._store_base_name = store_base_name

        self._sequence_id = sequence_id
        self._sequence_timestamp_interval_us = sequence_timestamp_interval_us
        self._generic_meta_data = generic_meta_data

        # Individual stores for each group are initialized lazily on-demand (indexed tar file or zarr directories)
        self._stores_rootgroups: dict[
            str, Tuple[nodes.Group, UPath]
        ] = {}  # maps component group names to stores, store path, and base groups
        self._store_type = store_type
        self._zarr_format = nodes.check_zarr_format(zarr_format)

        # registered component writers
        self._component_writers: dict[
            str, ComponentWriter
        ] = {}  # maps from component id to associated component writer

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    @property
    def sequence_timestamp_interval_us(self) -> HalfClosedInterval:
        return self._sequence_timestamp_interval_us

    @property
    def generic_meta_data(self) -> Dict[str, types.JsonLike]:
        return self._generic_meta_data

    @property
    def zarr_format(self) -> int:
        """The on-disk zarr format of the written component stores"""
        return self._zarr_format

    def get_base_group(self, component_group_name: Optional[str]) -> nodes.Group:
        """Lazily initializes ncore base-groups and underlying stores on demand"""

        if component_group_name is None:
            # empty group name represents the default group
            component_group_name = ""

        if (store_rootgroup := self._stores_rootgroups.get(component_group_name)) is not None:
            # Store already initialized, return it's root group
            return store_rootgroup[0]

        # Store doesn't exist yet, create it
        self._output_dir_path.mkdir(parents=True, exist_ok=True)

        # always use 'ncore4' as store name prefix
        store_name = "ncore4"
        if len(component_group_name):
            # append group name as suffix to store name if given
            store_name += f"-{component_group_name}"

        store: stores.StoreLike
        if self._store_type == "itar":
            # container-based zarr stores <base-name>.<store_name>.zarr.itar
            store_path = self._output_dir_path / f"{self._store_base_name}.{store_name}.zarr.itar"
            store = stores.IndexedTarStore(store_path, mode="w")
        elif self._store_type == "directory":
            # directory-based zarr stores <base-name>.<store_name>.zarr
            store_path = self._output_dir_path / f"{self._store_base_name}.{store_name}.zarr"
            store = stores.open_directory_store(store_path, mode="w")
        else:
            raise ValueError(f"Unknown store type {self._store_type}")

        # Create root group in store, storing dataset associated meta-data to root
        root_group = stores.create_root_group(
            store,
            self._zarr_format,
            {
                "sequence_id": self._sequence_id,
                "sequence_timestamp_interval_us": {
                    "start": self._sequence_timestamp_interval_us.start,
                    "stop": self._sequence_timestamp_interval_us.stop,
                },
                "generic_meta_data": self._generic_meta_data,
                "version": VERSION,
                "component_group_name": component_group_name,
            },
        )

        # Create store / base-group mapping
        self._stores_rootgroups[component_group_name] = root_group, store_path

        return root_group

    # To be called after all data was written
    def finalize(self) -> List[UPath]:
        """Finalize all writers, store their generic data and meta data, and close all stores after consolidating their meta data.

        Returns a list of the store paths
        """
        # Finalize all writers
        for component_writer in self._component_writers.values():
            component_writer.finalize()

        # Write deferred component metadata and generic data of all writers
        for component_writer in self._component_writers.values():
            # Merge generic_meta_data with regular component's meta data
            (cw_group := component_writer._group).update_attrs(
                {**component_writer._component_meta_data, "generic_meta_data": component_writer._generic_meta_data},
            )

            # Write generic data arrays if any
            if component_writer._generic_data:
                gd_group = cw_group.require_group("generic_data")
                for name, array in component_writer._generic_data.items():
                    gd_group.create_array(name, array, chunks=_normalize_chunks(array.shape))

        # Make sure the stores are consolidated and closed
        ret = []
        for root_group, store_path in self._stores_rootgroups.values():
            store = stores.get_group_store(root_group)

            stores.consolidate_store(store)

            # Finish writing all files
            stores.close_store(store)

            ret.append(store_path)

        return ret

    def register_component_writer(
        self,
        component_writer_type: "Callable[Concatenate[nodes.Group, HalfClosedInterval, P], CW]",
        component_instance_name: str,
        group_name: Optional[str] = None,
        generic_meta_data: Optional[Dict[str, types.JsonLike]] = None,
        *args: "P.args",
        **kwargs: "P.kwargs",
    ) -> CW:
        """Instantiates a component writer for the given type and instance name.

        The extra positional and keyword arguments are forwarded to the writer
        constructor.  Because the first parameter is typed with
        :data:`~typing.ParamSpec` + :data:`~typing.Concatenate`, type checkers
        can infer the correct extra-argument signature from the concrete writer
        class that is passed in."""
        generic_meta_data = unpack_optional(generic_meta_data, default={})

        assert len(component_instance_name) > 0, "Component instance name must not be empty"

        # The signature uses Callable[Concatenate[...], CW] for type-safe kwargs
        # inference, but we also need access to ComponentWriter static class attributes.
        writer_cls = cast(Type[ComponentWriter], component_writer_type)

        # Create component name from component base name and component instance name
        component_base_name = writer_cls.get_component_name()
        component_id = f"{component_base_name}:{component_instance_name}"

        assert component_id not in self._component_writers, f"Component writer for {component_id} already registered"

        # Create the component in the requested group, separated by component base name
        component_group = (
            self.get_base_group(group_name).require_group(component_base_name).require_group(component_instance_name)
        )

        # Initialize writer instance
        self._component_writers[component_id] = (
            component_writer_instance := component_writer_type(
                component_group, self._sequence_timestamp_interval_us, *args, **kwargs
            )
        )

        # Prepare component meta-data and initial meta data
        component_meta_data: Dict[str, types.JsonLike] = {
            "component_name": component_base_name,
            "component_instance_name": component_instance_name,
            "component_version": writer_cls.get_component_version(),
        }

        component_writer_instance._component_meta_data = component_meta_data
        component_writer_instance._generic_meta_data = generic_meta_data

        return component_writer_instance


class SequenceComponentGroupsReader:
    """SequenceComponentReader manages data component groups for reading for NCore V4 / zarr data for a single NCore sequence"""

    @staticmethod
    def expand_component_group_paths(
        component_group_paths: List[UPath] | List[Path],
    ) -> List[UPath]:
        """Expands possible sequence meta-data files in the given list of component group
        paths to the actual component group paths they reference"""

        ret: List[UPath] = []

        for component_group_path in component_group_paths:
            component_group_path = UPath(component_group_path).absolute()

            if component_group_path.is_file() and component_group_path.suffix == ".json":
                # sequence meta-data file - load and expand
                with component_group_path.open("r") as f:
                    sequence_meta = SequenceMeta.from_dict(json.load(f))

                for component_store_info in sequence_meta.component_stores:
                    ret.append(component_group_path.parent / UPath(component_store_info.path))
            else:
                # direct component group path
                ret.append(component_group_path)

        return ret

    def __init__(
        self,
        component_group_paths: List[UPath] | List[Path],
        open_consolidated: bool = True,
        itar_index_tail_read_size: int = 1 << 20,  # 1 MiB default
        max_threads: int | None = None,
        node_cache_size: int | None = None,
    ):
        """Initialize a SequenceComponentReader for a virtual sequence represented by a list of components.

        Args:
            component_group_paths: Universal paths / URLs to component groups to load (which may include sequence meta-data files),
                                   which need to represent a *single* sequence
            open_consolidated: If 'True', pre-load per-component meta-data when opening the components.
                               This is advisable if component data is accessed from *non-local*
                               storage to prevent latencies introduced when accessing the data.
                               If the component data is available on fast *local* storage, disabling
                               this option can speed up initial load times.
            itar_index_tail_read_size: When loading component groups from .itar files, the size of the tail buffer to read in a
                               single I/O call to initialize the tar file index to minimize reads.
                               Default 1 MiB size fits typical use cases, but may be increased if the compressed
                               index size is expected to be larger or decreased if remote access chunk sizes are smaller.
            max_threads:       The maximum number of threads used to load the different components (if None,
                               use interpreter-default number of threads for a ThreadPoolExecutor)
            node_cache_size:   Number of opened zarr nodes (groups / arrays, including their parsed meta-data but not
                               their data) each component reader caches for repeated accesses (e.g., of frames).
                               None caches all accessed nodes for the lifetime of the reader (typically a few kB per
                               node), a positive number caches the most recently used nodes only, 0 disables caching.
        """

        if node_cache_size is not None and node_cache_size < 0:
            raise ValueError(f"node_cache_size needs to be None or non-negative, got {node_cache_size}")
        self._node_cache_size = node_cache_size

        component_group_upaths: List[UPath] = self.expand_component_group_paths(component_group_paths)

        assert len(component_group_upaths), "No component inputs provided"

        # Load component stores concurrently (to hide latency) and check for sequence consistency
        self._component_stores: Dict[str, Tuple[nodes.Group, UPath]] = {}  # use str as the generic path / URL type

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:

            def thread_load_component_store(component_store_upath: UPath) -> Tuple[nodes.Group, UPath]:
                """Thread-executed shard opening"""

                # Make sure paths are absolute at this point
                component_store_upath = component_store_upath.absolute()

                _logger.info(f"SequenceStoreReader: Loading component store {component_store_upath}")

                component_store: stores.StoreLike
                if component_store_upath.is_file():
                    if not component_store_upath.name.endswith(".zarr.itar"):
                        # not a supported file-based store format
                        raise RuntimeError(
                            f"Unsupported file-based store format {component_store_upath}, expected .zarr.itar"
                        )

                    component_store = stores.IndexedTarStore(
                        component_store_upath, mode="r", index_tail_read_size=itar_index_tail_read_size
                    )
                else:
                    component_store = stores.open_directory_store(component_store_upath, mode="r")

                return stores.open_store(component_store, open_consolidated), component_store_upath

            for future in concurrent.futures.as_completed(
                [
                    executor.submit(thread_load_component_store, component_store_upath)
                    for component_store_upath in component_group_upaths
                ]
            ):
                # Note: thread completion order is not relevant here
                component_root, component_store_path = future.result()

                # Check sequence compatibility
                component_root_version = component_root.attr_str("version")
                if component_root_version not in [VERSION]:
                    raise RuntimeError(
                        f"Can't load V4 component store {component_store_path} with incompatible data version {component_root_version}"
                    )

                component_store_sequence_id = component_root.attr_str("sequence_id")
                interval = component_root.attr_dict("sequence_timestamp_interval_us")
                component_store_sequence_timestamp_interval_us = HalfClosedInterval(
                    cast(int, interval["start"]), cast(int, interval["stop"])
                )
                component_store_generic_meta_data = component_root.attr_dict("generic_meta_data")

                if not self._component_stores:
                    self._sequence_id: str = component_store_sequence_id
                    self._sequence_timestamp_interval_us = component_store_sequence_timestamp_interval_us
                    self._generic_meta_data: Dict[str, types.JsonLike] = component_store_generic_meta_data
                    self._version: str = component_root_version

                if not self._sequence_id == component_store_sequence_id:
                    raise RuntimeError("Can't load component store from different sequences")
                if not self._sequence_timestamp_interval_us == component_store_sequence_timestamp_interval_us:
                    raise RuntimeError("Can't load component store with different sequence timestamp intervals")
                if not self._generic_meta_data == component_store_generic_meta_data:
                    raise RuntimeError("Can't load component store with different generic meta-data")
                if not self._version == component_root_version:
                    raise RuntimeError("Can't load shards from different data versions")

                component_group_name = component_root.attr_str("component_group_name")
                if component_group_name in self._component_stores:
                    raise RuntimeError(f"Component group {component_group_name} loaded multiple times")

                self._component_stores[component_group_name] = (component_root, component_store_path)

        # Check version-compatibility
        if self._version != VERSION:
            raise ValueError(f"Loading incompatible version {self._version}, supporting {VERSION} only")

    def reload_resources(self) -> None:
        """Trigger a reload of each itar store - useful to re-initialize file objects in multi-process settings"""
        for component_root, _ in self._component_stores.values():
            if isinstance(store := stores.get_group_store(component_root), stores.IndexedTarStore):
                store.reload_resources()

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    @property
    def sequence_timestamp_interval_us(self) -> HalfClosedInterval:
        return self._sequence_timestamp_interval_us

    @property
    def generic_meta_data(self) -> Dict[str, types.JsonLike]:
        return self._generic_meta_data

    @property
    def component_store_paths(self) -> List[UPath]:
        return [path for _, path in self._component_stores.values()]

    def open_component_readers(
        self,
        component_reader_type: Type[CR],
    ) -> Dict[str, CR]:
        """Instantiates all component readers for the given component of all associated stores, identified by the component instance names"""

        ret = {}

        for component_root_group, _ in self._component_stores.values():
            if (component_name := component_reader_type.get_component_name()) not in component_root_group:
                continue

            # instantiate a reader for each of the components
            for component_instance_name, component_group in component_root_group.group(component_name).groups():
                assert component_instance_name not in ret, (
                    f"Component instance {component_instance_name} encountered multiple times"
                )

                # check if the reader supports the component version
                if not component_reader_type.supports_component_version(component_group.attr_str("component_version")):
                    continue

                ret[component_instance_name] = component_reader_type(
                    component_instance_name, component_group.with_cache(nodes.NodeCache.create(self._node_cache_size))
                )

        return ret

    def get_sequence_meta(self) -> SequenceMeta:
        """Returns full sequence meta-data summary (json-serializable)"""

        # collect component names and instances per store and component
        component_stores_info: List[SequenceMeta.ComponentStoreMeta] = []
        for component_root_group, component_store_path in self._component_stores.values():
            components: Dict[str, Dict[str, SequenceMeta.ComponentInstanceMeta]] = {}
            for component_name, component in component_root_group.groups():
                # collect component names and instances
                component_instances: Dict[str, SequenceMeta.ComponentInstanceMeta] = {}
                for component_instance_name, component_instance in component.groups():
                    component_instances[component_instance_name] = SequenceMeta.ComponentInstanceMeta(
                        version=component_instance.attr_str("component_version"),
                        generic_meta_data=component_instance.attr_dict("generic_meta_data"),
                    )

                components[component_name] = component_instances

            component_stores_info.append(
                SequenceMeta.ComponentStoreMeta(
                    path=component_store_path.name,
                    md5=MD5Hasher.hash(component_store_path),
                    components=components,
                )
            )

        # combine with sequence-wide information
        return SequenceMeta(
            sequence_id=self._sequence_id,
            sequence_timestamp_interval_us=SequenceMeta.TimestampInterval(
                start=self._sequence_timestamp_interval_us.start,
                stop=self._sequence_timestamp_interval_us.stop,
            ),
            generic_meta_data=self._generic_meta_data,
            version=self._version,
            component_stores=component_stores_info,
        )


class ComponentWriter(ABC):
    """Base class for V4 component writers.

    Subclasses must implement :meth:`get_component_name` and
    :meth:`get_component_version`, and may override :meth:`finalize` to flush
    buffered data.  All timestamps stored by the writer must fall within the
    sequence's ``sequence_timestamp_interval_us`` time range (available as
    ``self._sequence_timestamp_interval_us``).
    """

    @staticmethod
    @abstractmethod
    def get_component_name() -> str:
        """Returns the base name of the component writer"""
        ...

    @staticmethod
    @abstractmethod
    def get_component_version() -> str:
        """Returns the version of the current component writer"""
        ...

    def __init__(self, component_group: nodes.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
        """Initializes a component writer targeting the given component group and sequence time interval"""

        self._group = component_group
        self._sequence_timestamp_interval_us = sequence_timestamp_interval_us

        # Initialized by the SequenceComponentGroupsWriter.register_component_writer()
        self._component_meta_data: Dict[str, types.JsonLike] = {}
        self._generic_data: Dict[str, np.ndarray] = {}
        self._generic_meta_data: Dict[str, types.JsonLike] = {}

    def set_generic_data(
        self,
        data: Dict[str, np.ndarray],
        meta_data: Optional[Dict[str, types.JsonLike]] = None,
    ) -> None:
        """Attach named generic data arrays (and optional metadata) to this component.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Named numpy arrays to store under the component's ``generic_data/`` group (overwrites
            existing arrays with the same name, if any).
        meta_data : Optional[Dict[str, types.JsonLike]]
            Keys provided here **replace** any init-time / previous ``generic_meta_data``
            keys set during component registration or earlier method calls.
        """

        # Merge generic data: new data overwrites existing data with the same name
        self._generic_data.update(data)

        if meta_data is not None:
            # Merge generic_meta_data: init-time / set_generic_data meta overwrites
            self._generic_meta_data.update(meta_data)

    def finalize(self) -> None:
        """Overwrite to perform final operations after all user-data was written.

        Called by :meth:`SequenceComponentGroupsWriter.finalize` before
        component metadata is persisted.  Override to flush buffered data
        (e.g., create zarr datasets/groups).
        """
        pass


class ComponentReader(ABC):
    """Base class for V4 component readers.

    Subclasses must implement :meth:`get_component_name` and
    :meth:`supports_component_version`.  The underlying zarr group is
    accessible as ``self._group``; component metadata is available via the
    :attr:`instance_name`, :attr:`component_version`, and
    :attr:`generic_meta_data` properties.

    Methods reading array data have asynchronous variants (``*_async``), which can be awaited concurrently (e.g.,
    via ``asyncio.gather``) to overlap their I/O and decompression, and which don't block the running event loop.
    Metadata (attributes, names, timestamps) is served from memory and only provided synchronously.
    """

    @staticmethod
    @abstractmethod
    def get_component_name() -> str:
        """Returns the base name of the current component"""

    @staticmethod
    @abstractmethod
    def supports_component_version(version: str) -> bool:
        """Returns true if the component version is supported by the reader"""

    def __init__(self, component_instance_name: str, component_group: nodes.Group) -> None:
        """Initializes a component reader for a given component instance name and group.

        Nodes of the component are accessed via the node cache of the group (sized by `SequenceComponentGroupsReader`'s
        `node_cache_size`).
        """
        self._instance_name = component_instance_name
        self._group = component_group

        # Preload component meta-data and generic data group (if existing)
        self._component_meta_data: Dict[str, types.JsonLike] = self._group.attrs
        self._generic_data_group: Optional[nodes.Group] = (
            self._group.group("generic_data") if "generic_data" in self._group else None
        )

    @property
    def instance_name(self) -> str:
        """The user-defined name that distinguishes this component instance from others of the same type."""
        return self._instance_name

    @property
    def component_version(self) -> str:
        """Returns the component version of the loaded component"""
        return cast(str, self._component_meta_data["component_version"])

    @property
    def generic_meta_data(self) -> Dict[str, types.JsonLike]:
        """Returns the generic meta data of the loaded component"""
        return cast(Dict[str, types.JsonLike], self._component_meta_data["generic_meta_data"])

    def has_generic_data(self, name: str) -> bool:
        """Returns True if a named generic data array exists on this component"""
        return name in self._generic_data_group if self._generic_data_group is not None else False

    def get_generic_data_names(self) -> List[str]:
        """Returns the list of all generic data array names on this component"""
        return self._generic_data_group.members() if self._generic_data_group is not None else []

    def get_generic_data(self, name: str) -> np.ndarray:
        """Returns a named generic data array from this component.

        Raises KeyError if the name does not exist.
        """
        return self._generic_data_array(name).read()

    async def get_generic_data_async(self, name: str) -> np.ndarray:
        """Asynchronous :meth:`get_generic_data`"""
        return await self._generic_data_array(name).read_async()

    def _generic_data_array(self, name: str) -> nodes.Array:
        if self._generic_data_group is None:
            raise KeyError("Component has no generic_data")
        if name in self._generic_data_group:
            return self._generic_data_group.array(name)
        raise KeyError(f"Generic data '{name}' not found. Available: {self.get_generic_data_names()}")


CW = TypeVar("CW", bound=ComponentWriter)
CR = TypeVar("CR", bound=ComponentReader)
P = ParamSpec("P")


def validate_frame_name(name: str) -> str:
    """Checks if the given name is a valid frame name (non-empty, no whitespace), returns it if valid, raises AssertionError otherwise"""
    assert len(name) and not name.isspace(), f"Frame '{name}' is invalid, must not be empty or contain whitespace"

    return name


class PosesComponent:
    """Represents a generic set of static / dynamic poses (rigid transformations) between named coordinate frames"""

    COMPONENT_NAME: str = "poses"

    class Writer(ComponentWriter):
        """Poses data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component'"""
            return PosesComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current component writer"""
            return "v1"

        def __init__(self, component_group: nodes.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
            """Initializes the current component writer targeting the given component group and sequence time interval"""
            super().__init__(component_group, sequence_timestamp_interval_us)

            self.data: Dict = {"static_poses": {}, "dynamic_poses": {}}

        def finalize(self) -> None:
            """Actually store the json-encoded pose data"""

            self._group.create_group("static_poses", attributes=self.data["static_poses"])
            self._group.create_group("dynamic_poses", attributes=self.data["dynamic_poses"])

        def store_static_pose(
            self,
            source_frame_id: str,
            target_frame_id: str,
            pose: npt.NDArray[np.floating],  #: Source-to-target SE3 transformation (float32/64, [4,4])
        ) -> "Self":
            """Store a static pose (rigid transformation) between two named coordinate frames.

            Makes sure the inverse transformation is not already stored."""

            # Sanity checks
            assert pose.shape == (4, 4)
            assert np.issubdtype(pose.dtype, np.floating), "Poses must be of float type"
            assert np.all(pose[3, :] == [0.0, 0.0, 0.0, 1.0]), "Invalid SE3 transformation"

            key = (validate_frame_name(source_frame_id), validate_frame_name(target_frame_id))
            inv_key = key[::-1]

            assert key not in self.data["static_poses"], f"Static pose {key} already exists"
            assert inv_key not in self.data["static_poses"], f"Inverse static pose {inv_key} already exists"

            self.data["static_poses"][str(key)] = {"pose": pose.tolist(), "dtype": str(pose.dtype)}

            return self

        def store_dynamic_pose(
            self,
            source_frame_id: str,
            target_frame_id: str,
            poses: npt.NDArray[np.floating],  #: Source-to-target SE3 transformation trajectory (float32/64, [N,4,4])
            timestamps_us: npt.NDArray[
                np.uint64
            ],  #: All source-to-target transformation timestamps of the trajectory (uint64, [N,])
            #: If 'True', require that the dynamic poses fully cover the sequence time range -
            # setting this to 'False' can result in non-interpolatable poses outside the sequence time range in
            # downstream applications and is not advisable unless necessary
            require_sequence_time_coverage: bool = True,
        ) -> "Self":
            """Store a trajectory of dynamic poses (time-dependent rigid transformations) between two named coordinate frames.

            Makes sure the inverse transformation is not already stored."""

            # Sanity / timestamp consistency checks
            assert poses.shape[1:] == (4, 4)
            assert np.issubdtype(poses.dtype, np.floating), "Poses must be of float type"
            assert np.all(poses[:, 3, :] == [0.0, 0.0, 0.0, 1.0]), "Invalid SE3 transformations"

            assert timestamps_us.ndim == 1
            assert timestamps_us.dtype == np.dtype("uint64")

            assert len(poses) == len(timestamps_us)

            assert len(poses) > 1, "At least two poses required for a dynamic pose trajectory to support interpolation"

            assert np.all(np.diff(timestamps_us) > 0), "Timestamps must be strictly increasing"

            assert (
                timestamps_us[0].item() in self._sequence_timestamp_interval_us
                and timestamps_us[-1].item() in self._sequence_timestamp_interval_us
            ), "Dynamic poses samples must be fully contained in the sequence time range"

            if require_sequence_time_coverage:
                assert timestamps_us[0] == self._sequence_timestamp_interval_us.start, (
                    "Dynamic poses must cover the full sequence time range - "
                    "the first timestamp is after the sequence start time"
                )
                assert timestamps_us[-1] == self._sequence_timestamp_interval_us.stop - 1, (
                    "Dynamic poses must cover the full sequence time range - "
                    "the last timestamp is before the sequence end time"
                )

            key = (validate_frame_name(source_frame_id), validate_frame_name(target_frame_id))
            inv_key = key[::-1]

            assert key not in self.data["dynamic_poses"], f"Dynamic poses {key} already exists"
            assert inv_key not in self.data["dynamic_poses"], f"Inverse dynamic poses {inv_key} already exists"

            self.data["dynamic_poses"][str(key)] = {
                "poses": poses.tolist(),
                "timestamps_us": timestamps_us.tolist(),
                "dtype": str(poses.dtype),
            }

            return self

    class Reader(ComponentReader):
        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return PosesComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"

        def get_static_poses(self) -> Generator[Tuple[Tuple[str, str], npt.NDArray[np.floating]]]:
            """Returns all static poses (rigid transformations) between named coordinate frames, if available"""

            static_poses = self._group.group("static_poses")
            for key in static_poses.attrs:
                static_pose = static_poses.attr_dict(key)
                yield ast.literal_eval(key), np.array(static_pose["pose"], dtype=cast(str, static_pose["dtype"]))

        def get_dynamic_poses(
            self,
        ) -> Generator[Tuple[Tuple[str, str], Tuple[npt.NDArray[np.floating], npt.NDArray[np.uint64]]]]:
            """Returns all dynamic poses (time-dependent rigid transformations) between named coordinate frames, if available"""

            dynamic_poses_group = self._group.group("dynamic_poses")
            for key in dynamic_poses_group.attrs:
                dynamic_poses = dynamic_poses_group.attr_dict(key)
                yield (
                    ast.literal_eval(key),
                    (
                        np.array(dynamic_poses["poses"], dtype=cast(str, dynamic_poses["dtype"])),
                        np.array(dynamic_poses["timestamps_us"], dtype=np.uint64),
                    ),
                )

        def get_static_pose(self, source_frame_id: str, target_frame_id: str) -> npt.NDArray[np.floating]:
            """Returns static pose (rigid transformation) between two named coordinate frames, if available"""

            key = str((validate_frame_name(source_frame_id), validate_frame_name(target_frame_id)))
            if key not in (static_poses := self._group.group("static_poses")).attrs:
                raise KeyError(f"Static pose {key} not found")

            return np.array(static_poses.attr_dict(key)["pose"], dtype=np.float64)

        def get_dynamic_pose(
            self, source_frame_id: str, target_frame_id: str
        ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.uint64]]:
            """Returns dynamic poses (time-dependent rigid transformations) between two named coordinate frames, if available"""

            key = str((validate_frame_name(source_frame_id), validate_frame_name(target_frame_id)))
            if key not in (dynamic_poses_group := self._group.group("dynamic_poses")).attrs:
                raise KeyError(f"Dynamic poses {key} not found")

            dynamic_poses = dynamic_poses_group.attr_dict(key)
            return np.array(dynamic_poses["poses"], dtype=np.float64), np.array(
                dynamic_poses["timestamps_us"], dtype=np.uint64
            )


class IntrinsicsComponent:
    """Sensor intrinsic calibration data component"""

    COMPONENT_NAME: str = "intrinsics"

    class Writer(ComponentWriter):
        """Sensor intrinsics data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current intrinsic calibration component"""
            return IntrinsicsComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current intrinsic calibration component"""
            return "v1"

        def __init__(self, component_group: nodes.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
            """Initializes the current component writer targeting the given component group and sequence time interval"""
            super().__init__(component_group, sequence_timestamp_interval_us)

            self._cameras_group = self._group.create_group("cameras")
            self._lidars_group = self._group.create_group("lidars")

        def store_camera_intrinsics(
            self,
            camera_id: str,
            # intrinsics
            camera_model_parameters: types.CameraModelParameters,
        ) -> "Self":
            """Store camera-associated intrinsics"""

            # Prepare meta-data containing the serialization of the mandatory camera model / optional external distortion parameters

            meta_data = types.encode_camera_model_parameters(camera_model_parameters)

            self._cameras_group.create_group(camera_id, attributes=meta_data)

            return self

        def store_lidar_intrinsics(
            self,
            lidar_id: str,
            # intrinsics
            lidar_model_parameters: types.LidarModelParameters,
        ) -> "Self":
            """Store lidar-associated intrinsics"""

            # Prepare meta-data containing the serialization of the mandatory lidar model
            meta_data = types.encode_lidar_model_parameters(lidar_model_parameters)

            self._lidars_group.create_group(lidar_id, attributes=meta_data)

            return self

    class Reader(ComponentReader):
        """Sensor intrinsics data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return IntrinsicsComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"

        def get_camera_model_parameters(self, camera_id: str) -> types.CameraModelParameters:
            """Returns the camera model associated with the requested camera sensor"""
            return types.decode_camera_model_parameters(self._group.group(f"cameras/{camera_id}").attrs)

        def get_lidar_model_parameters(self, lidar_id: str) -> Optional[types.LidarModelParameters]:
            """Returns the lidar model associated with the requested lidar sensor"""
            lidars_group = self._group.group("lidars")

            if lidar_id not in lidars_group:
                return None

            return types.decode_lidar_model_parameters(self._group.group(f"lidars/{lidar_id}").attrs)


class MasksComponent:
    """Sensor masks data component"""

    COMPONENT_NAME: str = "masks"

    class Writer(ComponentWriter):
        """Sensor masks data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current sensor masks component"""
            return MasksComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current sensor masks component"""
            return "v1"

        def __init__(self, component_group: nodes.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
            """Initializes the current component writer targeting the given component group and sequence time interval"""
            super().__init__(component_group, sequence_timestamp_interval_us)

            self._cameras_group = self._group.create_group("cameras")

        def store_camera_masks(
            self,
            camera_id: str,
            # named camera sensor masks
            mask_images: Dict[str, PILImage.Image],
        ) -> "Self":
            """Store camera-associated masks"""

            # Store mask names
            camera_grp = self._cameras_group.create_group(
                camera_id, attributes={"mask_names": list(mask_images.keys())}
            )

            # Store mask images
            for mask_name, mask_image in mask_images.items():
                with io.BytesIO() as buffer:
                    FORMAT = "png"
                    mask_image.save(buffer, format=FORMAT, optimize=True)  # encodes as png
                    # store mask data (uncompressed, as already encoded)
                    camera_grp.create_bytes_array(mask_name, buffer.getvalue(), attributes={"format": FORMAT})

            return self

    class Reader(ComponentReader):
        """Sensor masks data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return MasksComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"

        def get_camera_mask_names(self, camera_id: str) -> List[str]:
            """Returns all constant camera mask names"""

            camera_group = self._group.group(f"cameras/{camera_id}")
            return (
                [str(name) for name in camera_group.attr_list("mask_names")]
                if "mask_names" in camera_group.attrs
                else []
            )

        def get_camera_mask_image(self, camera_id: str, mask_name: str) -> PILImage.Image:
            """Returns constant named camera mask image"""

            mask_dataset = self._group.array(f"cameras/{camera_id}/{mask_name}")

            return PILImage.open(
                io.BytesIO(mask_dataset.read_bytes()),
                formats=[mask_dataset.attr_str("format")],
            )

        async def get_camera_mask_image_async(self, camera_id: str, mask_name: str) -> PILImage.Image:
            """Asynchronous :meth:`get_camera_mask_image`"""

            mask_dataset = self._group.array(f"cameras/{camera_id}/{mask_name}")

            return PILImage.open(
                io.BytesIO(await mask_dataset.read_bytes_async()),
                formats=[mask_dataset.attr_str("format")],
            )

        def get_camera_mask_images(self, camera_id: str) -> Generator[Tuple[str, PILImage.Image]]:
            """Returns all constant named camera mask images"""

            for mask_name in self.get_camera_mask_names(camera_id):
                yield mask_name, self.get_camera_mask_image(camera_id, mask_name)


class BaseSensorComponentWriter(ComponentWriter):
    """Base class for all sensor component writers"""

    def __init__(self, component_group: nodes.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
        """Initializes the current component writer targeting the given component group and sequence time interval"""
        super().__init__(component_group, sequence_timestamp_interval_us)

        self._frames_group = self._group.create_group("frames")

        self._frames_timestamps_us: Dict[
            int, int
        ] = {}  # collect end-of-frame timestamps mapping to start of frame timestamps

    def finalize(self) -> None:
        """Perform final operations after all user-data was written to the sensor component"""

        # Collect all frame timestamps to be stored as global property (supporting no frames at all and out-of-order frames)
        frames_timestamps_us = np.array(
            [(self._frames_timestamps_us[end], end) for end in sorted(self._frames_timestamps_us.keys())],
            dtype=np.uint64,
        ).reshape((-1, 2))

        # Validate all start/end-of-frame timestamps to be monotonically increasing
        assert np.all(frames_timestamps_us[:-1, 0] < frames_timestamps_us[1:, 0]), (
            "Start of frame timestamps are not monotonically increasing"
        )
        assert np.all(frames_timestamps_us[:-1, 1] < frames_timestamps_us[1:, 1]), (
            "End of frame timestamps are not monotonically increasing"
        )

        # Store as meta-data of frames group
        self._frames_group.update_attrs({"frames_timestamps_us": frames_timestamps_us.tolist()})

    def _get_frame_group(
        self,
        # end-of-frame timestamp, or start-of-frame / end-of-frame timestamps
        timestamps_us: Union[int, npt.NDArray[np.uint64]],
    ) -> nodes.Group:
        """Returns the group of a frame, initializing it if required"""

        if isinstance(timestamps_us, np.ndarray):
            frame_id = cast(np.ndarray, timestamps_us)[1].item()  # end-of-frame timestamp is frame ID
        else:
            frame_id = timestamps_us

        return self._frames_group.require_group(str(frame_id))

    def _store_base_frame(
        self,
        # start-of-frame / end-of-frame timestamps
        frame_timestamps_us: npt.NDArray[np.uint64],
        # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
        generic_data: Dict[str, npt.NDArray[Any]],
        generic_meta_data: Dict[str, types.JsonLike],
    ) -> nodes.Group:
        # Sanity / timestamp consistency checks
        assert np.shape(frame_timestamps_us) == (2,)
        assert frame_timestamps_us.dtype == np.dtype("uint64")
        assert frame_timestamps_us[1] >= frame_timestamps_us[0]

        assert frame_timestamps_us[0].item() in self._sequence_timestamp_interval_us, (
            "Frame start timestamp must be contained in the sequence time range"
        )
        assert frame_timestamps_us[1].item() in self._sequence_timestamp_interval_us, (
            "Frame end timestamp must be contained in the sequence time range"
        )

        # Initialize frame group
        frame_group = self._get_frame_group(frame_timestamps_us)

        # Store timestamp data
        assert frame_timestamps_us[1].item() not in self._frames_timestamps_us, (
            "Frame with the same end-of-frame timestamp already exists"
        )
        self._frames_timestamps_us[frame_timestamps_us[1].item()] = frame_timestamps_us[0].item()

        # Store additional generic frame data and meta-data (not dimension / dtype checked)
        frame_generic_data_group = frame_group.create_group("generic_data", attributes=generic_meta_data)
        for name, value in generic_data.items():
            frame_generic_data_group.create_array(
                name,
                value,
                # we are not accessing sub-ranges, so disable chunking
                chunks=_normalize_chunks(value.shape),
                compression=_FAST_DECODE_COMPRESSION,
            )

        return frame_group


class BaseSensorComponentReader(ComponentReader):
    """Base class for all sensor component readers"""

    def __init__(self, component_instance_name: str, component_group: nodes.Group) -> None:
        """Initializes a component reader for a given component instance name and group"""
        super().__init__(component_instance_name, component_group)

        if "frames" not in self._group:
            raise RuntimeError("Sensor component doesn't contain any frames")

        # preload frame timestamps and create map
        self._frames_timestamps_us = np.array(
            self._group.group("frames").attr_list("frames_timestamps_us"), dtype=np.uint64
        )
        self._frame_end_to_frame_timestamps_us = {
            end: np.array([self._frames_timestamps_us[i, 0], end], dtype=np.uint64)
            for i, end in enumerate(self._frames_timestamps_us[:, 1])
        }

    @staticmethod
    def _frame_id(timestamps_us: Union[int, npt.NDArray[np.uint64]]) -> int:
        """Returns the frame ID (end-of-frame timestamp) of end-of-frame / start-of-frame + end-of-frame timestamps"""
        if isinstance(timestamps_us, np.ndarray):
            return cast(np.ndarray, timestamps_us)[1].item()
        return timestamps_us

    def _get_frame_group(
        self,
        # end-of-frame timestamp, or start-of-frame / end-of-frame timestamps
        timestamps_us: Union[int, npt.NDArray[np.uint64]],
    ) -> nodes.Group:
        """Returns the group of a frame"""

        return self._group.group("frames").group(str(self._frame_id(timestamps_us)))

    @property
    def frames_timestamps_us(self) -> npt.NDArray[np.uint64]:
        return np.array(self._group.group("frames").attr_list("frames_timestamps_us"), dtype=np.uint64)

    @property
    def frames_count(self) -> int:
        return len(self._frames_timestamps_us)

    def get_frame_timestamps_us(self, timestamp_us: int) -> npt.NDArray[np.uint64]:
        return self._frame_end_to_frame_timestamps_us[timestamp_us]

    # Generic per-frame data
    def get_frame_generic_data_names(self, timestamp_us: int) -> List[str]:
        """List of all generic frame-data names"""

        return self._get_frame_group(timestamp_us).group("generic_data").members()

    def has_frame_generic_data(self, timestamp_us: int, name: str) -> bool:
        """Signals if named generic frame-data exists"""

        return name in self.get_frame_generic_data_names(timestamp_us)

    def get_frame_generic_data(self, timestamp_us: int, name: str) -> npt.NDArray[Any]:
        """Returns generic frame-data for a specific frame and name"""

        return self._get_frame_group(timestamp_us).group("generic_data").array(name).read()

    async def get_frame_generic_data_async(self, timestamp_us: int, name: str) -> npt.NDArray[Any]:
        """Asynchronous :meth:`get_frame_generic_data`"""

        return await self._get_frame_group(timestamp_us).group("generic_data").array(name).read_async()

    def get_frame_generic_meta_data(self, timestamp_us: int) -> Dict[str, types.JsonLike]:
        """Returns generic frame meta-data for a specific frame"""

        return self._get_frame_group(timestamp_us).group("generic_data").attrs


class BaseRayBundleSensorComponentWriter(BaseSensorComponentWriter):
    """Base class for all ray bundle sensor component writers"""

    def _store_frame_ray_bundle(
        self,
        # start-of-frame / end-of-frame timestamps
        frame_timestamps_us: npt.NDArray[np.uint64],
        # number of rays and returns per ray
        n_rays: int,
        n_returns: int,
        # per-ray data components with N leading dimension along with chunk specifiers
        ray_data: Dict[str, Tuple[npt.NDArray[Any], Tuple[int, ...]]],
        # per-return data components with (R, N) leading dimension along with chunk specifiers. Non-existing values are indicated via NaNs
        return_data: Dict[str, Tuple[npt.NDArray[np.float32], Tuple[int, ...]]],
    ) -> None:
        ## Initialize ray bundle group
        frame_group = self._get_frame_group(frame_timestamps_us)
        ray_bundle_group = frame_group.create_group("ray_bundle", attributes={"n_rays": n_rays})

        # Store per-ray data
        for name, (ray_data_data, chunks) in ray_data.items():
            assert len(ray_data_data) == n_rays, f"{name} doesn't have required ray count"
            ray_bundle_group.create_array(
                name,
                ray_data_data,
                chunks=_normalize_chunks(chunks),
                compression=_FAST_DECODE_COMPRESSION,
            )

        ## Initialize ray bundle returns group
        ray_bundle_returns_group = frame_group.create_group("ray_bundle_returns", attributes={"n_returns": n_returns})

        # Store per-return data
        absent_mask = None
        for name, (return_data_data, chunks) in return_data.items():
            assert return_data_data.shape[:2] == (n_returns, n_rays), (
                f"{name} doesn't have required ray / return count {(n_returns, n_rays)}"
            )

            # Determine local absent mask from NaN values,
            # which needs to be consistent over all dimensions of a return
            local_absent_mask = np.isnan(return_data_data)
            if return_data_data.ndim > 2:
                # reduce over all additional dimensions and check for consistency
                d_axes = tuple(range(2, return_data_data.ndim))
                all_nan = local_absent_mask.all(axis=d_axes)
                any_nan = local_absent_mask.any(axis=d_axes)
                assert np.array_equal(all_nan, any_nan), (
                    f"Partial NaN detected at positions: {np.argwhere(all_nan != any_nan)[:5].tolist()} in higher-dimensional return data {name}"
                )
                local_absent_mask = all_nan

            assert local_absent_mask.shape == (n_returns, n_rays), (
                f"Invalid NaN mask shape {local_absent_mask.shape} for return data {name}"
            )

            if absent_mask is None:
                # initialize absent mask from first return data
                absent_mask = local_absent_mask
            else:
                # validate absent mask consistency
                assert np.array_equal(absent_mask, local_absent_mask), f"Inconsistent NaN masks in return data {name}"

            ray_bundle_returns_group.create_array(
                name,
                return_data_data,
                chunks=_normalize_chunks(chunks),
                compression=_FAST_DECODE_COMPRESSION,
            )

        if absent_mask is None:
            # Initialize empty absent mask if no per-return data was provided
            absent_mask = np.full((n_returns, n_rays), fill_value=False, dtype=bool)

        valid_mask_packed = np.packbits(~absent_mask)

        frame_group.create_array(
            "ray_bundle_returns_valid_mask_packed",
            valid_mask_packed,
            # we are not accessing sub-ranges, so disable chunking
            chunks=_normalize_chunks(valid_mask_packed.shape),
            compression=_FAST_DECODE_COMPRESSION,
            attributes={"n_returns": n_returns, "n_rays": n_rays},
        )


class BaseRayBundleSensorComponentReader(BaseSensorComponentReader):
    """Base class for all ray bundle sensor component readers"""

    # per-ray data
    def _get_ray_bundle_group(self, timestamp_us: int) -> nodes.Group:
        """Returns the ray bundle group of a frame"""

        return self._get_frame_group(timestamp_us).group("ray_bundle")

    def get_frame_ray_bundle_count(self, timestamp_us: int) -> int:
        """Returns the number of rays and ray returns for a specific frame"""

        return self._get_ray_bundle_group(timestamp_us).attr_int("n_rays")

    def get_frame_ray_bundle_data_names(self, timestamp_us: int) -> List[str]:
        """List of all ray bundle data names for a frame"""

        return self._get_ray_bundle_group(timestamp_us).members()

    def has_frame_ray_bundle_data(self, timestamp_us: int, name: str) -> bool:
        """Signals if named ray bundle data exists for a frame"""

        return name in self._get_ray_bundle_group(timestamp_us)

    def get_frame_ray_bundle_data(self, timestamp_us: int, name: str) -> npt.NDArray[Any]:
        """Returns named ray bundle data for a frame"""

        return self._get_frame_group(timestamp_us).group("ray_bundle").array(name).read()

    async def get_frame_ray_bundle_data_async(self, timestamp_us: int, name: str) -> npt.NDArray[Any]:
        """Asynchronous :meth:`get_frame_ray_bundle_data`"""

        return await self._get_frame_group(timestamp_us).group("ray_bundle").array(name).read_async()

    # per-ray return data
    def _get_ray_bundle_returns_group(self, timestamp_us: int) -> nodes.Group:
        """Returns the ray bundle returns group of a frame"""

        return self._get_frame_group(timestamp_us).group("ray_bundle_returns")

    def get_frame_ray_bundle_return_count(self, timestamp_us: int) -> int:
        """Returns the number of ray returns for a specific frame"""

        return self._get_ray_bundle_returns_group(timestamp_us).attr_int("n_returns")

    def get_frame_ray_bundle_return_data_names(self, timestamp_us: int) -> List[str]:
        """List of all ray bundle return data names for a frame"""

        return self._get_ray_bundle_returns_group(timestamp_us).members()

    def has_frame_ray_bundle_return_data(self, timestamp_us: int, name: str) -> bool:
        """Signals if named ray bundle return data exists for a frame"""

        return name in self._get_ray_bundle_returns_group(timestamp_us)

    def get_frame_ray_bundle_return_valid_mask(self, timestamp_us: int) -> npt.NDArray[np.bool_]:
        """Returns the per-ray return valid mask for a frame"""

        valid_mask_packed = self._get_frame_group(timestamp_us).array("ray_bundle_returns_valid_mask_packed")
        return self._unpack_valid_mask(valid_mask_packed, valid_mask_packed.read())

    async def get_frame_ray_bundle_return_valid_mask_async(self, timestamp_us: int) -> npt.NDArray[np.bool_]:
        """Asynchronous :meth:`get_frame_ray_bundle_return_valid_mask`"""

        valid_mask_packed = self._get_frame_group(timestamp_us).array("ray_bundle_returns_valid_mask_packed")
        return self._unpack_valid_mask(valid_mask_packed, await valid_mask_packed.read_async())

    @staticmethod
    def _unpack_valid_mask(valid_mask_packed: nodes.Array, packed: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        n_returns, n_rays = valid_mask_packed.attr_int("n_returns"), valid_mask_packed.attr_int("n_rays")

        return np.unpackbits(packed, count=n_returns * n_rays).astype(np.bool_).reshape((n_returns, n_rays))

    def get_frame_ray_bundle_return_data(
        self, timestamp_us: int, name: str, return_index: Optional[int]
    ) -> npt.NDArray[np.float32]:
        """Returns named ray bundle return data for a frame, optionally indexed by return index to accelerate data-retrieval"""

        # only references the underlying Array, don't load it's data yet
        return_array = self._get_ray_bundle_returns_group(timestamp_us).array(name)

        if return_index is None:
            return return_array.read()  # load all returns
        else:
            return return_array.read((return_index, Ellipsis))  # load specific return only

    async def get_frame_ray_bundle_return_data_async(
        self, timestamp_us: int, name: str, return_index: Optional[int]
    ) -> npt.NDArray[np.float32]:
        """Asynchronous :meth:`get_frame_ray_bundle_return_data`"""

        return_array = self._get_ray_bundle_returns_group(timestamp_us).array(name)

        if return_index is None:
            return await return_array.read_async()
        else:
            return await return_array.read_async((return_index, Ellipsis))


class CameraSensorComponent:
    """Camera sensor data component"""

    COMPONENT_NAME: str = "cameras"

    class Writer(BaseSensorComponentWriter):
        """Camera sensor data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current camera sensor component"""
            return CameraSensorComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current camera sensor component"""
            return "v1"

        def store_frame(
            self,
            # image data
            image_binary_data: bytes,
            image_format: str,
            # start-of-frame / end-of-frame timestamps
            frame_timestamps_us: npt.NDArray[np.uint64],
            # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
            generic_data: Dict[str, npt.NDArray[Any]],
            generic_meta_data: Dict[str, types.JsonLike],
        ) -> "Self":
            # Initialize frame
            frame_group = self._store_base_frame(frame_timestamps_us, generic_data, generic_meta_data)

            # Store image data (uncompressed, as already encoded)
            frame_group.create_bytes_array("image", image_binary_data, attributes={"format": image_format})

            return self

    class Reader(BaseSensorComponentReader):
        """Camera sensor data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return CameraSensorComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"

        class EncodedImageDataHandle:
            """References encoded image data without loading it"""

            def __init__(self, image_dataset: nodes.Array) -> None:
                self._image_dataset = image_dataset

            def get_data(self) -> types.EncodedImageData:
                """Loads the referenced encoded image data to memory"""
                return types.EncodedImageData(
                    self._image_dataset.read_bytes(),
                    self._image_dataset.attr_str("format"),
                )

            async def get_data_async(self) -> types.EncodedImageData:
                """Asynchronous :meth:`get_data`"""
                return types.EncodedImageData(
                    await self._image_dataset.read_bytes_async(),
                    self._image_dataset.attr_str("format"),
                )

        def get_frame_handle(self, timestamp_us: int) -> EncodedImageDataHandle:
            """Returns the frame's encoded image data"""
            return self.EncodedImageDataHandle(self._get_frame_group(timestamp_us).array("image"))

        def get_frame_data(self, timestamp_us: int) -> types.EncodedImageData:
            """Returns the frame's encoded image data"""
            return self.get_frame_handle(timestamp_us).get_data()

        async def get_frame_data_async(self, timestamp_us: int) -> types.EncodedImageData:
            """Asynchronous :meth:`get_frame_data`"""
            return await self.get_frame_handle(timestamp_us).get_data_async()

        def get_frame_image(self, timestamp_us: int) -> PILImage.Image:
            """Returns the frame's decoded image data"""
            return self.get_frame_data(timestamp_us).get_decoded_image()


class LidarSensorComponent:
    """Lidar sensor data component"""

    COMPONENT_NAME: str = "lidars"

    class Writer(BaseRayBundleSensorComponentWriter):
        """Lidar sensor data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current lidar sensor component"""
            return LidarSensorComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current lidar sensor component"""
            return "v1"

        def store_frame(
            self,
            # ray-associated data for N rays (data common to all returns)
            direction: npt.NDArray[
                np.float32
            ],  # per-ray unit-norm direction vectors in sensor frame at measure time (raw / not motion-compensated, needs to have unit norm) (float32, [N, 3])
            timestamp_us: npt.NDArray[np.uint64],  # per-ray timestamp in microseconds (uint64, [N])
            model_element: Optional[
                npt.NDArray[np.uint16]
            ],  # per-ray model element indices, if applicable (uint16, [N, 2])
            # per-ray return data for R returns - non-existing values are indicated via NaNs
            distance_m: npt.NDArray[
                np.float32
            ],  # per-point metric distance along the ray at measure time time (raw / not motion-compensated, needs to be non-negative) (float32, [R, N])
            intensity: npt.NDArray[np.float32],  # per-point intensity normalized to [0.0, 1.0] range (float32, [R, N])
            # start-of-frame / end-of-frame timestamps
            frame_timestamps_us: npt.NDArray[np.uint64],
            # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
            generic_data: Dict[str, npt.NDArray[Any]],
            generic_meta_data: Dict[str, types.JsonLike],
        ) -> "Self":
            ## Sanity / consistency checks
            assert direction.ndim == 2
            assert np.shape(direction)[1] == 3
            assert direction.dtype == np.dtype("float32")

            # make sure directions are unit-norm
            assert np.all(np.abs(np.sum(direction**2, axis=1) - 1.0) < 1e-4), "Direction vectors are not unit-norm"

            n_rays = len(direction)

            ## Initialize frame
            self._store_base_frame(frame_timestamps_us, generic_data, generic_meta_data)

            ## Per frame data
            ray_data: Dict[str, Tuple[npt.NDArray[Any], Tuple[int, ...]]] = {
                "direction": (direction, direction.shape),
            }

            assert timestamp_us.dtype == np.dtype("uint64")
            assert timestamp_us.shape == (n_rays,)
            if n_rays:
                assert (frame_timestamps_us[0] <= timestamp_us.min()) and (
                    timestamp_us.max() <= frame_timestamps_us[1]
                ), "Point timestamps outside frame time bounds"
            ray_data["timestamp_us"] = (timestamp_us, timestamp_us.shape)

            if model_element is not None:
                assert model_element.shape == (n_rays, 2)
                assert model_element.dtype == np.dtype("uint16")
                ray_data["model_element"] = (model_element, model_element.shape)

            ## Per return data

            return_data: Dict[str, Tuple[npt.NDArray[np.float32], Tuple[int, ...]]] = {}

            # distance
            assert distance_m.ndim == 2
            n_returns = distance_m.shape[0]
            assert distance_m.shape[1] == n_rays
            assert distance_m.dtype == np.dtype("float32")
            distance_m_finite = distance_m[np.isfinite(distance_m)]
            assert np.all(distance_m_finite >= 0.0), "Distance contains negative values"
            return_data["distance_m"] = (distance_m, (1, n_rays))  # chunk along returns

            # intensity
            assert intensity.shape == (n_returns, n_rays)
            assert intensity.dtype == np.dtype("float32")
            intensity_finite = intensity[np.isfinite(intensity)]
            assert np.all(0.0 <= intensity_finite) and np.all(intensity_finite <= 1.0), "Intensity not normalized"
            return_data["intensity"] = (intensity, (1, n_rays))  # chunk along returns

            # Store point-clouds data
            self._store_frame_ray_bundle(frame_timestamps_us, n_rays, n_returns, ray_data, return_data)

            return self

    class Reader(BaseRayBundleSensorComponentReader):
        """Lidar sensor data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return LidarSensorComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"


class RadarSensorComponent:
    """Radar sensor data component"""

    COMPONENT_NAME: str = "radars"

    class Writer(BaseRayBundleSensorComponentWriter):
        """Radar sensor data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current radar sensor component"""
            return RadarSensorComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current radar sensor component"""
            return "v1"

        def store_frame(
            self,
            # ray-associated data for N rays (data common to all returns)
            direction: npt.NDArray[
                np.float32
            ],  # per-point unit-norm direction vectors in sensor frame at measure time (raw / not motion-compensated, needs to have unit norm) (float32, [N, 3])
            timestamp_us: npt.NDArray[np.uint64],  # per-point timestamp in microseconds (uint64, [N])
            # per-ray return data for R returns - non-existing values are indicated via NaNs
            distance_m: npt.NDArray[
                np.float32
            ],  # per-point metric distance along the ray at measure time time (raw / not motion-compensated, needs to be non-negative) (float32, [R, N])
            # start-of-frame / end-of-frame timestamps
            frame_timestamps_us: npt.NDArray[np.uint64],
            # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
            generic_data: Dict[str, npt.NDArray[Any]],
            generic_meta_data: Dict[str, types.JsonLike],
        ) -> "Self":
            ## Sanity / consistency checks
            assert direction.ndim == 2
            assert np.shape(direction)[1] == 3
            assert direction.dtype == np.dtype("float32")

            # make sure directions are unit-norm
            assert np.all(np.abs(np.sum(direction**2, axis=1) - 1.0) < 1e-4), "Direction vectors are not unit-norm"

            n_rays = len(direction)

            ## Initialize frame
            self._store_base_frame(frame_timestamps_us, generic_data, generic_meta_data)

            ## Per frame data
            ray_data: Dict[str, Tuple[npt.NDArray[Any], Tuple[int, ...]]] = {
                "direction": (direction, direction.shape),
            }

            assert timestamp_us.dtype == np.dtype("uint64")
            assert timestamp_us.shape == (n_rays,)
            if n_rays:
                assert (frame_timestamps_us[0] <= timestamp_us.min()) and (
                    timestamp_us.max() <= frame_timestamps_us[1]
                ), "Point timestamps outside frame time bounds"
            ray_data["timestamp_us"] = (timestamp_us, timestamp_us.shape)

            ## Per return data

            return_data: Dict[str, Tuple[npt.NDArray[np.float32], Tuple[int, ...]]] = {}

            # distance
            assert distance_m.ndim == 2
            n_returns = distance_m.shape[0]
            assert distance_m.shape[1] == n_rays
            assert distance_m.dtype == np.dtype("float32")
            distance_m_finite = distance_m[np.isfinite(distance_m)]
            assert np.all(distance_m_finite >= 0.0), "Distance contains negative values"
            return_data["distance_m"] = (distance_m, (1, n_rays))  # chunk along returns

            # Store point-clouds data
            self._store_frame_ray_bundle(frame_timestamps_us, n_rays, n_returns, ray_data, return_data)

            return self

    class Reader(BaseRayBundleSensorComponentReader):
        """Radar sensor data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return RadarSensorComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"


class CuboidsComponent:
    """Data component representing cuboid track observations"""

    COMPONENT_NAME: str = "cuboids"

    class Writer(ComponentWriter):
        """Cuboid track observations component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current lidar sensor component"""
            return CuboidsComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current lidar sensor component"""
            return "v1"

        def store_observations(
            self,
            cuboid_observations: List[types.CuboidTrackObservation],  # individual observation
        ) -> "Self":
            obs_dict_list = []
            for obs in cuboid_observations:
                # Check timestamp validity before serialization
                assert obs.timestamp_us in self._sequence_timestamp_interval_us, (
                    f"Cuboid track observation timestamp {obs.timestamp_us} not in the sequence time range"
                )
                assert obs.reference_frame_timestamp_us in self._sequence_timestamp_interval_us, (
                    f"Cuboid track observation reference frame timestamp {obs.reference_frame_timestamp_us} not in the sequence time range"
                )
                obs_dict_list.append(obs.to_dict())

            self._group.create_group("cuboids", attributes={"cuboid_track_observations": obs_dict_list})

            return self

    class Reader(ComponentReader):
        """Cuboid tracks component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return CuboidsComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "v1"

        def get_observations(self) -> Generator[types.CuboidTrackObservation]:
            """Returns all stored cuboid track observations"""

            for obs in self._group.group("cuboids").attr_list("cuboid_track_observations"):
                yield types.CuboidTrackObservation.from_dict(cast(Dict[str, types.JsonLike], obs))


class PointCloudsComponent:
    """Data component representing unstructured point clouds with optional typed attributes."""

    COMPONENT_NAME: str = "point_clouds"

    @dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
    class AttributeSchema(dataclasses_json.DataClassJsonMixin):
        """Schema for a single per-point attribute (frozen / hashable).

        Serialization uses :class:`~dataclasses_json.DataClassJsonMixin`:

        * ``transform_type`` is stored as its uppercase enum name
          (``"INVARIANT"``, ``"DIRECTION"``, ``"POINT"``).
        * ``dtype`` is stored as a numpy dtype string (e.g. ``"float32"``).
        * ``shape_suffix`` is stored as a JSON array of ints.
        """

        transform_type: PointCloud.AttributeTransformType = util.enum_field(PointCloud.AttributeTransformType)
        dtype: np.dtype = util.dtype_field()
        shape_suffix: Tuple[int, ...] = field(
            default=(),
            metadata=dataclasses_json.config(encoder=list, decoder=tuple),
        )

    # --------------------------------------------------------------------------
    # Writer
    # --------------------------------------------------------------------------

    class Writer(ComponentWriter):
        """Point-clouds component writer."""

        @staticmethod
        def get_component_name() -> str:
            return PointCloudsComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            return "v1"

        def __init__(
            self,
            component_group: nodes.Group,
            sequence_timestamp_interval_us: HalfClosedInterval,
            coordinate_unit: PointCloud.CoordinateUnit,
            attribute_schemas: Optional[Dict[str, PointCloudsComponent.AttributeSchema]] = None,
        ) -> None:
            super().__init__(component_group, sequence_timestamp_interval_us)
            attribute_schemas = unpack_optional(attribute_schemas, default={})

            self._coordinate_unit = coordinate_unit
            self._attribute_schemas = attribute_schemas
            self._pc_timestamps: List[int] = []

            # Pre-create the pcs group
            self._pcs_group = self._group.require_group("pcs")
            self._pcs_group.update_attrs(
                {
                    "coordinate_unit": coordinate_unit.name,
                    "attribute_schemas": {name: s.to_dict() for name, s in self._attribute_schemas.items()},
                },
            )

        def store_pc(
            self,
            xyz: npt.NDArray[np.float32],
            reference_frame_id: str,
            reference_frame_timestamp_us: int,
            attributes: Optional[Dict[str, npt.NDArray[Any]]] = None,
            generic_data: Optional[Dict[str, npt.NDArray[Any]]] = None,
            generic_meta_data: Optional[Dict[str, types.JsonLike]] = None,
        ) -> None:
            """Store a single point cloud.

            Attributes represent per-point data with a schema declared in the component-level ``attribute_schemas`` meta-data.
            Generic data and meta-data allow storing additional per-point-cloud information without a predefined schema.

            The ``reference_frame_timestamp_us`` is also collected into the
            source-level ``pc_timestamps_us`` array written by :meth:`finalize`.
            """
            attributes = unpack_optional(attributes, default={})
            generic_data = unpack_optional(generic_data, default={})
            generic_meta_data = unpack_optional(generic_meta_data, default={})
            # -- Validate xyz --
            assert xyz.dtype == np.dtype("float32")
            assert xyz.ndim == 2 and xyz.shape[1] == 3, f"xyz must be (N, 3), got {xyz.shape}"
            N = xyz.shape[0]

            # -- Validate timestamp --
            assert reference_frame_timestamp_us in self._sequence_timestamp_interval_us, (
                f"reference_frame_timestamp_us {reference_frame_timestamp_us} not in sequence time range"
            )

            # -- Validate attributes against schema --
            assert (provided_keys := set(attributes.keys())) == (schema_keys := set(self._attribute_schemas.keys())), (
                f"Attribute keys mismatch: expected {schema_keys}, got {provided_keys}"
            )

            for attr_name, attr_array in attributes.items():
                schema = self._attribute_schemas[attr_name]
                expected_shape = (N,) + schema.shape_suffix
                assert attr_array.shape == expected_shape, (
                    f"Attribute '{attr_name}' shape mismatch: expected {expected_shape}, got {attr_array.shape}"
                )
                assert np.dtype(attr_array.dtype) == schema.dtype, (
                    f"Attribute '{attr_name}' dtype mismatch: expected {schema.dtype}, got {attr_array.dtype}"
                )

            # -- Create per-pc group --
            # Store per-pc metadata (timestamp lives in source-level pc_timestamps_us array)
            pc_group = self._pcs_group.create_group(
                str(len(self._pc_timestamps)),
                attributes={
                    "reference_frame_id": reference_frame_id,
                    "generic_meta_data": generic_meta_data,
                },
            )

            # Store xyz
            pc_group.create_array("xyz", xyz, chunks=_normalize_chunks(xyz.shape, require_nonzero_dims=(1,)))

            # Store schema-declared attributes
            for attr_name, attr_array in attributes.items():
                pc_group.create_array(attr_name, attr_array, chunks=_normalize_chunks(attr_array.shape))

            # Store generic data
            gd_group = pc_group.create_group("generic_data")
            for gd_name, gd_array in generic_data.items():
                gd_group.create_array(gd_name, gd_array, chunks=_normalize_chunks(gd_array.shape))

            self._pc_timestamps.append(reference_frame_timestamp_us)

        def finalize(self) -> None:
            """Write pc_timestamps_us array (derived from per-pc reference_frame_timestamp_us values)."""
            ts_array = np.array(self._pc_timestamps, dtype=np.uint64)
            self._group.create_array("pc_timestamps_us", ts_array, chunks=_normalize_chunks(ts_array.shape))

    # --------------------------------------------------------------------------
    # Reader
    # --------------------------------------------------------------------------

    class Reader(ComponentReader):
        """Point-clouds component reader."""

        @staticmethod
        def get_component_name() -> str:
            return PointCloudsComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            return version == "v1"

        def __init__(self, component_instance_name: str, component_group: nodes.Group) -> None:
            super().__init__(component_instance_name, component_group)

            self._pc_timestamps_us: npt.NDArray[np.uint64] = np.array(self._group.array("pc_timestamps_us").read())

            pcs_group = self._group.group("pcs")
            self._coordinate_unit = PointCloud.CoordinateUnit[pcs_group.attr_str("coordinate_unit")]
            self._attribute_schemas: Dict[str, PointCloudsComponent.AttributeSchema] = {
                name: PointCloudsComponent.AttributeSchema.from_dict(cast(Dict[str, types.JsonLike], s))
                for name, s in pcs_group.attr_dict("attribute_schemas").items()
            }

        # -- properties --------------------------------------------------------

        @property
        def coordinate_unit(self) -> PointCloud.CoordinateUnit:
            return self._coordinate_unit

        @property
        def pcs_count(self) -> int:
            return len(self._pc_timestamps_us)

        @property
        def pc_timestamps_us(self) -> "npt.NDArray[np.uint64]":
            return self._pc_timestamps_us

        @property
        def attribute_names(self) -> List[str]:
            return list(self._attribute_schemas.keys())

        # -- schema access -----------------------------------------------------

        def get_attribute_schema(self, name: str) -> PointCloudsComponent.AttributeSchema:
            assert name in self._attribute_schemas, f"Unknown attribute: {name}"
            return self._attribute_schemas[name]

        # -- per-pc data access ------------------------------------------------

        def _pc_group(self, pc_index: int) -> nodes.Group:
            return self._group.group(f"pcs/{pc_index}")

        def get_pc_xyz(self, pc_index: int) -> npt.NDArray[np.float32]:
            return self._group.array(f"pcs/{pc_index}/xyz").read()

        async def get_pc_xyz_async(self, pc_index: int) -> npt.NDArray[np.float32]:
            """Asynchronous :meth:`get_pc_xyz`"""
            return await self._group.array(f"pcs/{pc_index}/xyz").read_async()

        def get_pc_attribute(self, pc_index: int, name: str) -> npt.NDArray[Any]:
            return self._group.array(f"pcs/{pc_index}/{name}").read()

        async def get_pc_attribute_async(self, pc_index: int, name: str) -> npt.NDArray[Any]:
            """Asynchronous :meth:`get_pc_attribute`"""
            return await self._group.array(f"pcs/{pc_index}/{name}").read_async()

        def get_pc_reference_frame_id(self, pc_index: int) -> str:
            return str(self._pc_group(pc_index).attrs["reference_frame_id"])

        def get_pc_reference_frame_timestamp_us(self, pc_index: int) -> int:
            return int(self._pc_timestamps_us[pc_index])

        def get_pc_generic_data_names(self, pc_index: int) -> List[str]:
            return self._group.group(f"pcs/{pc_index}/generic_data").members()

        def has_pc_generic_data(self, pc_index: int, name: str) -> bool:
            return name in self._group.group(f"pcs/{pc_index}/generic_data")

        def get_pc_generic_data(self, pc_index: int, name: str) -> npt.NDArray[Any]:
            return np.asarray(self._group.array(f"pcs/{pc_index}/generic_data/{name}").read())

        async def get_pc_generic_data_async(self, pc_index: int, name: str) -> npt.NDArray[Any]:
            """Asynchronous :meth:`get_pc_generic_data`"""
            return await self._group.array(f"pcs/{pc_index}/generic_data/{name}").read_async()

        def get_pc_generic_meta_data(self, pc_index: int) -> Dict[str, types.JsonLike]:
            pc_group = self._pc_group(pc_index)
            return pc_group.attr_dict("generic_meta_data") if "generic_meta_data" in pc_group.attrs else {}


class CameraLabelsComponent:
    """Data component for storing per-camera image-aligned labels (depth, segmentation, flow, etc.)."""

    COMPONENT_NAME: str = "camera_labels"

    # --------------------------------------------------------------------------
    # Writer
    # --------------------------------------------------------------------------

    class Writer(ComponentWriter):
        """Camera-labels component writer."""

        @staticmethod
        def get_component_name() -> str:
            return CameraLabelsComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            return "v1"

        def __init__(
            self,
            component_group: nodes.Group,
            sequence_timestamp_interval_us: HalfClosedInterval,
            descriptor: types.CameraLabelDescriptor,
        ) -> None:
            super().__init__(component_group, sequence_timestamp_interval_us)

            self._descriptor = descriptor

            # Initialize labels group and timestamps list
            self._labels_group = self._group.require_group("labels")
            self._labels_group.update_attrs({"descriptor": descriptor.to_dict()})
            self._timestamps: List[int] = []

        def store_label(
            self,
            data: "Union[npt.NDArray[Any], bytes]",
            timestamp_us: int,
            generic_meta_data: Optional[Dict[str, types.JsonLike]] = None,
        ) -> None:
            """Store a single camera label.

            Parameters
            ----------
            data
                 For RAW encoding: a numpy array of shape ``(H, W)`` or ``(H, W, *schema.shape_suffix)``
                 in the logical dtype (``schema.dtype``).  When quantization is configured, the
                 data is automatically quantized to the on-disk integer representation.
                 For IMAGE_ENCODED encoding: raw image bytes in the schema.encoded_format.
            timestamp_us
                Timestamp in microseconds - must fall within the sequence interval.
            generic_meta_data
                Optional per-label metadata.
            """
            generic_meta_data = unpack_optional(generic_meta_data, default={})

            # Sanity checks
            assert timestamp_us in self._sequence_timestamp_interval_us, (
                f"timestamp_us {timestamp_us} not in sequence time range"
            )
            assert timestamp_us not in self._timestamps, f"Duplicate timestamp_us: {timestamp_us}"

            # Store label-associated data in a dedicated subgroup named by the timestamp
            label_group = self._labels_group.create_group(str(timestamp_us))

            if self._descriptor.label_schema.encoding == types.LabelEncoding.RAW:
                assert isinstance(data, np.ndarray), "RAW encoding requires a numpy array"

                # Validate shape: (H, W) for scalar, (H, W, *shape_suffix) for multi-channel
                if self._descriptor.label_schema.shape_suffix:
                    assert data.ndim == 2 + len(self._descriptor.label_schema.shape_suffix), (
                        f"Expected ndim={2 + len(self._descriptor.label_schema.shape_suffix)}, got {data.ndim}"
                    )
                    assert data.shape[2:] == self._descriptor.label_schema.shape_suffix, (
                        f"shape_suffix mismatch: expected {self._descriptor.label_schema.shape_suffix}, got {data.shape[2:]}"
                    )
                else:
                    assert data.ndim == 2, f"Scalar label must be 2-D (H, W), got ndim={data.ndim}"

                # Validate dtype — caller must pass data in the expected dtype
                assert np.dtype(data.dtype) == self._descriptor.label_schema.dtype, (
                    f"dtype mismatch: expected {self._descriptor.label_schema.dtype}, got {data.dtype}"
                )

                # Quantize if configured
                stored = data
                if (q := self._descriptor.label_schema.quantization) is not None:
                    raw: np.ndarray = data
                    stored = np.round((raw.astype(q.intermediate_dtype) - q.offset) / q.scale).astype(q.quantized_dtype)

                label_group.create_array(
                    "data", stored, chunks=_normalize_chunks(stored.shape, require_nonzero_dims=(0, 1))
                )

            elif self._descriptor.label_schema.encoding == types.LabelEncoding.IMAGE_ENCODED:
                assert isinstance(data, bytes), "IMAGE_ENCODED encoding requires bytes"

                label_group.create_array(
                    "data",
                    np.frombuffer(data, dtype=np.uint8),
                    compression=None,
                    attributes={"format": self._descriptor.label_schema.encoded_format},
                )

            else:
                raise ValueError(f"Unsupported label encoding: {self._descriptor.label_schema.encoding}")

            label_group.update_attrs({"generic_meta_data": generic_meta_data})

            self._timestamps.append(timestamp_us)

        def finalize(self) -> None:
            """Write sorted timestamps_us array."""
            ts_array = np.array(sorted(self._timestamps), dtype=np.uint64)
            self._group.create_array("timestamps_us", ts_array, chunks=_normalize_chunks(ts_array.shape))

    # --------------------------------------------------------------------------
    # Reader
    # --------------------------------------------------------------------------

    class Reader(ComponentReader):
        """Camera-labels component reader."""

        @staticmethod
        def get_component_name() -> str:
            return CameraLabelsComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            return version == "v1"

        def __init__(self, component_instance_name: str, component_group: nodes.Group) -> None:
            super().__init__(component_instance_name, component_group)

            self._timestamps_us: npt.NDArray[np.uint64] = self._group.array("timestamps_us").read()
            self._timestamp_to_index: Dict[int, int] = {int(ts): i for i, ts in enumerate(self._timestamps_us)}
            self._descriptor = types.CameraLabelDescriptor.from_dict(
                self._group.group("labels").attr_dict("descriptor")
            )

        # -- properties --------------------------------------------------------

        @property
        def label_descriptor(self) -> types.CameraLabelDescriptor:
            return self._descriptor

        @property
        def labels_count(self) -> int:
            return len(self._timestamps_us)

        @property
        def timestamps_us(self) -> npt.NDArray[np.uint64]:
            return self._timestamps_us

        # -- per-label access --------------------------------------------------

        def _label_group(self, timestamp_us: int) -> nodes.Group:
            assert timestamp_us in self._timestamp_to_index, (
                f"Unknown timestamp: {timestamp_us}. Available: {list(self._timestamp_to_index.keys())[:5]}..."
            )
            return self._group.group(f"labels/{timestamp_us}")

        class CameraLabelHandle:
            """References label data without eagerly loading it.

            Implements the :class:`CameraLabelHandleProtocol` protocol, providing access to the label
            data, schema, timestamp, and per-label metadata.
            """

            def __init__(
                self,
                label_group: nodes.Group,
                descriptor: types.CameraLabelDescriptor,
                timestamp_us: int,
                generic_meta_data: Dict[str, types.JsonLike],
            ) -> None:
                self._label_group = label_group
                self._descriptor = descriptor
                self._timestamp_us = timestamp_us
                self._generic_meta_data = generic_meta_data

            @property
            def descriptor(self) -> types.CameraLabelDescriptor:
                return self._descriptor

            @property
            def timestamp_us(self) -> int:
                return self._timestamp_us

            @property
            def generic_meta_data(self) -> Dict[str, types.JsonLike]:
                return self._generic_meta_data

            def get_data(self) -> npt.NDArray[Any]:
                """Load and return the label data as a numpy array.

                For RAW encoding, applies de-quantization if specified in the schema.
                For IMAGE_ENCODED encoding, decodes the image bytes via PIL.
                """
                self._check_encoding()
                return self._decode(self._label_group.array("data").read())

            async def get_data_async(self) -> npt.NDArray[Any]:
                """Asynchronous :meth:`get_data`"""
                self._check_encoding()
                return self._decode(await self._label_group.array("data").read_async())

            def _check_encoding(self) -> None:
                if self._descriptor.label_schema.encoding not in (
                    types.LabelEncoding.RAW,
                    types.LabelEncoding.IMAGE_ENCODED,
                ):
                    raise ValueError(f"Unsupported label encoding: {self._descriptor.label_schema.encoding}")

            def _decode(self, data: npt.NDArray[Any]) -> npt.NDArray[Any]:
                if self._descriptor.label_schema.encoding == types.LabelEncoding.RAW:
                    # De-quantize if configured
                    if (q := self._descriptor.label_schema.quantization) is not None:
                        data = (data.astype(q.intermediate_dtype) * q.scale + q.offset).astype(
                            self._descriptor.label_schema.dtype
                        )
                    return data

                image = PILImage.open(io.BytesIO(data.tobytes()))
                return np.asarray(image, dtype=self._descriptor.label_schema.dtype)

            def get_encoded_data(self) -> Optional[bytes]:
                """Return the raw encoded bytes for IMAGE_ENCODED labels, or None for RAW."""
                if self._descriptor.label_schema.encoding == types.LabelEncoding.IMAGE_ENCODED:
                    return self._label_group.array("data").read().tobytes()
                return None

            async def get_encoded_data_async(self) -> Optional[bytes]:
                """Asynchronous :meth:`get_encoded_data`"""
                if self._descriptor.label_schema.encoding == types.LabelEncoding.IMAGE_ENCODED:
                    return (await self._label_group.array("data").read_async()).tobytes()
                return None

        def get_label(self, timestamp_us: int) -> CameraLabelHandle:
            """Return a lazy handle to the label data at the given timestamp."""
            label_group = self._label_group(timestamp_us)
            return self.CameraLabelHandle(
                label_group, self._descriptor, timestamp_us, label_group.attr_dict("generic_meta_data")
            )
