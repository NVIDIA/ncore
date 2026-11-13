.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _installation:

Installation
============

The NCore Python APIs are provided as ``pip``-installable wheels published to
`PyPI <https://pypi.org/project/nvidia-ncore>`_ under the package name
``nvidia-ncore``:

.. code-block:: bash

   pip install nvidia-ncore

``nvidia-ncore`` supports Python 3.8 and above. It works with zarr-python 2 and zarr-python 3 (see
:ref:`zarr_versions`); zarr-python 3 requires Python 3.11 or above, and is installed by default there (pin
``"zarr<3"`` to keep zarr-python 2). Writing zarr format v3 stores requires zarr-python 3.

Pre-releases are only installed with ``--pre`` or an exact version:

.. code-block:: bash

   pip install --pre nvidia-ncore
