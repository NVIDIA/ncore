.. Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

.. _data_overview : 


Overview
========

NCore provides a general canonical data specification supporting sensor data
recordings from various sources.

Access to the data is provided via a common and simple to use API, which enables
development and research applications to consume different dataset types in a
consistent way.

The canonical data format supports both NV-internal AV-data (``Hyperion 8`` /
future-variants) and robotics (``Carter`` platform) recordings, as well as
3rdparty datasets like the `Waymo Open Datasets <https://waymo.com/open/>`_. 

NCore uses the :ref:`Version 4 (V4) <v4-data-format>` *component-based* format,
which enables modular, independently-managed data components with enhanced
flexibility and scalability.

The coordinate system conventions are described in :ref:`data_conventions`,
while detailed format specifications are provided in :ref:`data_formats`.
