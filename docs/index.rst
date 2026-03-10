.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

NCore
=====

NCore provides a canonical data specification, APIs, and tools for multi-sensor
recordings in robotics and autonomous vehicle applications. The format supports
NV-specific AV data (``NVIDIA Hyperion 8/8.1`` and future variants), robotics
platforms (``NVIDIA Carter``), as well as 3rd-party datasets like the
`Waymo Open Dataset <https://waymo.com/open/>`_.

NCore's latest :ref:`V4 component-based data format <v4-data-format>` enables
modular, independently-managed generic data components with flexible composition
and scalability.

The project is developed within the `NVIDIA SIL Lab <https://research.nvidia.com/labs/sil/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Data Representation

   data/conventions
   data/sensor_models
   data/formats

.. toctree::
   :maxdepth: 1
   :caption: APIs

   apis/install
   apis/ncore

.. toctree::
   :maxdepth: 2
   :caption: Tools

   tools/data_vis
   tools/ncore_vis
   tools/ncore_sequence_meta
   conversions/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :name: tutorials

   tutorial/data_loading
   tutorial/data_sanity_check
