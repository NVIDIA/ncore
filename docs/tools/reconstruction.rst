.. Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

Reconstruction
==============

Data stored in NRECore-specific dataformats can be used as the input to scene reconstruction tools provided by the SDK

Poisson Surface Reconstruction
------------------------------

The tool ``//scripts:ncore_surface_rec`` performs point-cloud-based Poisson-based surface reconstruction and
exports the results to an output folder.

Example invocation::

    bazel run //scripts:ncore_surface_rec \
        -- \
        --shard-file-pattern=<SHARD_FILE_PATTERN> \
        --output-dir=<OUTPUT_FOLDER>

.. figure:: surfacerec.png
   :figwidth: 60%
   :width: 80%

   Surface reconstruction example
