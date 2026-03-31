.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _storage_and_access:


Storage and Access
==================

NCore V4 component stores (see :ref:`data_formats`) can be persisted in two
storage formats and accessed from local or remote storage backends.

.. _itar_store_format:

Indexed Tar Archive Format (``.itar``)
--------------------------------------

Each component group is a `zarr <https://zarr.readthedocs.io/en/stable/>`_
group stored either as a directory-based ``.zarr`` store or as a single-file
``.zarr.itar`` (indexed tar) archive. The itar format packages all zarr chunks
as sequential tar members in a single file and appends a compressed index at
the end, combining the streaming efficiency of tar with random-access
capability.

.. figure:: itar.svg
   :width: 100%

   Comparison of regular tar files (as used by
   `WebDataset <https://github.com/webdataset/webdataset>`_, supporting fast
   linear streaming but no random access) with the indexed tar format, which
   appends a compressed index enabling O(1) key lookups and direct seeks to
   any chunk.

The itar store implements the zarr ``Store`` interface, so it can be used as a
drop-in replacement for directory stores in all NCore APIs. Via
`UPath <https://github.com/fsspec/universal_pathlib>`_,
itar containers can also be accessed transparently from cloud storage backends
(e.g., S3, GCS) without requiring a local copy.

**Tradeoffs:**

* **itar** (single file) -- efficient for distribution, cloud storage, and
  atomic transfers; supports both sequential streaming and random access via
  the appended index
* **directory store** -- individual chunk files on disk; simpler for debugging
  and incremental updates

Both formats are accessed through the same
:class:`~ncore.data.v4.SequenceComponentGroupsReader` and
:class:`~ncore.data.v4.SequenceComponentGroupsWriter` APIs.

Loading V4 Data
---------------

V4 sequences are loaded by specifying one or more component store paths:

.. code-block:: python

   from ncore.data.v4 import SequenceComponentGroupsReader
   from pathlib import Path
   
   # Load sequence from multiple component stores
   reader = SequenceComponentGroupsReader([
       Path("ncore4.zarr.itar"),           # default components
       Path("ncore4-calibv2.zarr.itar"),   # alternative calibration
   ])
   
   # Access specific components
   poses_readers = reader.open_component_readers(PosesComponent.Reader)
   camera_readers = reader.open_component_readers(CameraSensorComponent.Reader)

.. _cloud_storage_access:

Cloud and Remote Storage Access
-------------------------------

NCore accesses all data paths through
`UPath <https://github.com/fsspec/universal_pathlib>`_ (``universal_pathlib``),
a drop-in ``pathlib.Path`` replacement built on top of
`fsspec <https://filesystem-spec.readthedocs.io/>`_. This means component
stores can be read transparently from cloud storage backends -- the same
:class:`~ncore.data.v4.SequenceComponentGroupsReader` API works for local
files and remote URLs alike.

Supported URL Schemes
~~~~~~~~~~~~~~~~~~~~~

Any protocol that fsspec supports can be used as a component store path.
Common examples:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Protocol
     - Example URL
   * - **S3**
     - ``s3://my-bucket/sequences/seq01/ncore4.zarr.itar``
   * - **GCS**
     - ``gs://my-bucket/sequences/seq01/ncore4.zarr.itar``
   * - **Azure Blob**
     - ``az://my-container/sequences/seq01/ncore4.zarr.itar``
   * - **HTTP(S)**
     - ``https://example.com/data/ncore4.zarr.itar``
   * - **Local**
     - ``/data/sequences/seq01/ncore4.zarr.itar``

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

``nvidia-ncore`` ships with ``universal_pathlib`` (and its transitive
dependency ``fsspec``), which is sufficient for local paths. To access
remote storage you need to install the corresponding fsspec filesystem
implementation:

.. list-table::
   :header-rows: 1
   :widths: 15 20 30

   * - Protocol
     - Extra package
     - Credentials / configuration
   * - S3
     - `s3fs <https://s3fs.readthedocs.io/>`_
     - AWS credentials (``~/.aws/credentials``, env vars, or IAM role)
   * - GCS
     - `gcsfs <https://gcsfs.readthedocs.io/>`_
     - ``GOOGLE_APPLICATION_CREDENTIALS`` or ``gcloud auth``
   * - Azure Blob
     - `adlfs <https://github.com/fsspec/adlfs>`_
     - ``AZURE_STORAGE_CONNECTION_STRING`` or ``az login``
   * - HTTP(S)
     - *(built-in)*
     - n/a

Install the extra package for the protocol you need, for example:

.. code-block:: bash

   pip install nvidia-ncore s3fs          # for S3
   pip install nvidia-ncore gcsfs         # for GCS
   pip install nvidia-ncore adlfs         # for Azure Blob

Loading Remote Component Stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass remote URLs directly to
:class:`~ncore.data.v4.SequenceComponentGroupsReader`:

.. code-block:: python

   from ncore.data.v4 import SequenceComponentGroupsReader
   from upath import UPath

   reader = SequenceComponentGroupsReader([
       UPath("s3://my-bucket/sequences/seq01/ncore4.zarr.itar"),
       UPath("s3://my-bucket/sequences/seq01/ncore4-labels.zarr.itar"),
   ])

For **S3-compatible endpoints** or when a specific **AWS profile** is needed,
pass additional keyword arguments through ``UPath``:

.. code-block:: python

   # Use a named AWS profile
   store_path = UPath(
       "s3://my-bucket/sequences/seq01/ncore4.zarr.itar",
       profile="my-aws-profile",
   )

   # Tune download performance
   store_path = UPath(
       "s3://my-bucket/sequences/seq01/ncore4.zarr.itar",
       profile="my-aws-profile",
       default_block_size=50 * 1024 * 1024,   # 50 MB download chunks
       default_cache_type="readahead",          # fsspec file-descriptor caching strategy
   )

   # Point to an S3-compatible endpoint (e.g. MinIO)
   store_path = UPath(
       "s3://my-bucket/sequences/seq01/ncore4.zarr.itar",
       client_kwargs={"endpoint_url": "https://minio.example.com"},
   )

   reader = SequenceComponentGroupsReader([store_path])

All keyword arguments accepted by the underlying
`S3FileSystem <https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem>`_
(or the respective fsspec filesystem class for other protocols) can be
forwarded this way.

Performance Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Use the** ``.itar`` **format** for cloud-stored data. The indexed tar archive
  enables random access with a single file, avoiding the large number of
  small HTTP requests that directory-based zarr stores would incur.

* **Enable consolidated metadata** (the default). The ``open_consolidated``
  parameter on :class:`~ncore.data.v4.SequenceComponentGroupsReader` is
  ``True`` by default, which pre-loads all zarr metadata in a single read.
  This is especially important for remote stores where each metadata
  lookup would otherwise be a separate round-trip.

* **Increase the block size** for high-bandwidth connections. The
  ``default_block_size`` parameter on ``UPath`` controls how much data
  is fetched per request. Larger values (e.g. 50--100 MB) reduce the
  number of requests at the cost of higher per-request latency.

* **Consider local caching** for repeated access to the same data. fsspec
  supports transparent caching via the ``simplecache`` or ``filecache``
  protocols:

  .. code-block:: python

     # Cache remote files locally on first access
     store_path = UPath(
         "simplecache::s3://my-bucket/sequences/seq01/ncore4.zarr.itar",
         s3={"profile": "my-aws-profile"},
         simplecache={"cache_storage": "/tmp/ncore_cache"},
     )
