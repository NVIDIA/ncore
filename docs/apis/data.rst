.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

``ncore.data`` Package
======================

.. automodule:: ncore.data
   :members:

.. toctree::
   :maxdepth: 1

   data.v4
   data.stores

JSON Types
----------

.. autodata:: ncore.data.Json

   A JSON value as read (e.g., attributes of nodes): of concrete types, dictionaries and lists.

.. autodata:: ncore.data.JsonDocument

   A JSON document as read (a dictionary at the top level).

.. autodata:: ncore.data.JsonLike

   A JSON-like value to write (e.g., attributes of nodes): of any mappings / sequences of JSON-like values (e.g.,
   read-only mappings, tuples), written as JSON objects / arrays. JSON values are JSON-like.

.. autodata:: ncore.data.JsonDocumentLike

   A JSON-like document to write (a mapping at the top level, see :data:`JsonLike`).
