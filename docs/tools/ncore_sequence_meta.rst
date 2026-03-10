.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Sequence Metadata
=================

The tool ``//tools:ncore_sequence_meta`` extracts comprehensive metadata from an
NCore sequences and outputs these as JSON file that references the component
stores. The output includes the sequence ID, timestamp range, component store
paths with MD5 checksums, component versions, and generic metadata fields.

This is useful when a data converter did not produce a metadata file, or when
additional components have been added to an existing sequence and the metadata
file needs to be regenerated to reflect all component stores.

Usage
-----

Basic invocation::

    bazel run //tools:ncore_sequence_meta \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        v4 \
        --component-group=<SEQUENCE_META.json>

With multiple component groups (e.g., after extending a sequence with new
components)::

    bazel run //tools:ncore_sequence_meta \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>

Options
-------

**Global options:**

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--output-dir``
     - (required)
     - Directory for the output JSON file
   * - ``--output-file``
     - ``<sequence_id>.json``
     - Custom output filename
   * - ``--open-consolidated / --no-open-consolidated``
     - enabled
     - Pre-load consolidated zarr metadata for faster access
   * - ``--debug``
     - off
     - Enable debug-level logging

**V4 sub-command options:**

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--component-group``
     - (required)
     - Component group or sequence meta path (repeatable)
   * - ``--poses-component-group``
     - ``default``
     - Component group name for poses
   * - ``--intrinsics-component-group``
     - ``default``
     - Component group name for intrinsics
   * - ``--masks-component-group``
     - ``default``
     - Component group name for masks (``none`` to disable)
   * - ``--cuboids-component-group``
     - ``default``
     - Component group name for cuboids (``none`` to disable)
