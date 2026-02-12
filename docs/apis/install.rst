.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
.. SPDX-License-Identifier: Apache-2.0

.. _installation: 

Wheel Installation
==================

The NCore python APIs are provided as ``pip``-installable wheels maintained in PyPi at `package registry <https://test.pypi.org/manage/project/nvidia-ncore>`_.

The wheels are called ``nvidia-ncore`` and can be installed by:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nvidia-ncore

For authentication, this requires a one-time setup of the gitlab personal 
access token (see `link <https://github.com/NVIDIA/ncore/blob/main/CONTRIBUTING.md#setup-gitlab-personal-access-token--docker-credentials>`_ for details).

Currently, ``nvidia-ncore`` wheels support all python version starting from ``python3.8``.
