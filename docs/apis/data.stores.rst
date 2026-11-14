.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

``ncore.data.stores`` Package
=============================

.. automodule:: ncore.data.stores
   :members:
   :exclude-members: ZarrStore

Types
-----

.. autodata:: ncore.data.stores.ZarrFormat

   zarr formats (on-disk format versions of zarr hierarchies).

.. autodata:: ncore.data.stores.Selection

   Basic selections of array data (integers, slices and Ellipsis).

.. autodata:: ncore.data.stores.FillValue

   Fill values of arrays (read for elements of chunks not stored): numpy scalars (e.g., ``numpy.integer``,
   ``numpy.floating``, ``numpy.bool_``, ``numpy.bytes_``, ``numpy.str_``), or Python scalars converted to the array's
   dtype.

.. autodata:: ncore.data.stores.Attributes

   Attributes of nodes to write (written as plain JSON, read back as :data:`ncore.data.JsonDocument`).

.. autodata:: ncore.data.stores.IndexTailRetention

   Which index tail read of an indexed tar archive (the end of the archive read when opening it, see
   ``index_tail_read_size`` of :class:`IndexedTarArchive`) is kept in memory once its hierarchy is opened:

   - ``"release_if_consolidated"``: release the tail read if the hierarchy was opened from its consolidated metadata
     (which then holds all node metadata), keep it otherwise (node metadata records read when accessing nodes may lie
     within it)
   - ``"release_after_opening"``: release the tail read after opening the hierarchy
   - ``"keep_complete_archives"``: release the tail read after opening the hierarchy, unless it holds the whole archive
     (small archives stay in memory)
   - ``"keep"``: keep the tail read (records near the end of the archive are served from memory for its lifetime)

   See :meth:`Store.release_index_tail` and :meth:`IndexedTarArchive.release_index_tail`.

.. autodata:: ncore.data.stores.BloscCname

   Blosc compressors (of :class:`BloscCompression`).

.. autodata:: ncore.data.stores.BloscShuffle

   Blosc shuffle modes (of :class:`BloscCompression`).

.. autodata:: ncore.data.stores.ZarrStore
   :no-value:

   The store base class of the installed zarr-python version (zarr-python 2: ``zarr._storage.store.BaseStore``,
   zarr-python 3: ``zarr.abc.store.Store``), see :class:`ZarrStoreAdapter`.
