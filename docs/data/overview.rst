.. Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

.. _data_overview : 


Overview
========

NRECore-SDK provides a general canonical data specification for sensor data recordings from various sources.


Access to the data is provided via a common and simple to use API to facilitate development and research applications to consume different dataset types.

The canonical data format supports both NV-internal AV-data (``Hyperion 8`` / future-variants) and robotics (``Carter`` platform) recordings,
as well as 3rdparty datasets like the `Waymo Open Datasets <https://waymo.com/open/>`_. 

The latest specification of NRECore datasets is  :ref:`Version-3 <v3-data-format>`, which implements a cloud-optimized storage format of dataset sequences partitioned into self-contained data shards.

A specification of the canonical data conventions and data-specific properties is given in :ref:`Conventions <data_conventions>`.
