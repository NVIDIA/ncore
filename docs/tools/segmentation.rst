.. Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

.. note:: 
   Usage of the inference models is *approved* for internal usage, but the data the models were trained on **is not commercially viable**

Segmentations
=============

Data stored in NCore-specific dataformats can be used as the input to different inference models provided as tools by the SDK

Semantic and Instance Segmentations
-----------------------------------

The tool ``//scripts:ncore_extract_segmentation`` performs both semantic- (``--semantic-seg``) and instance-segmentation (``--instance-seg``) for stored camera frames,
exporting the results to an output folder.

Example invocation::

    bazel run //scripts:ncore_extract_segmentation \
        -- \
        --shard-file-pattern=<SHARD_FILE_PATTERN> \
        --output-dir=<OUTPUT_FOLDER> \
        --semantic-seg \
        --instance-seg

.. figure:: semantic-seg.png
   :figwidth: 50%
   :width: 80%

   Semantic image segmentation example

.. figure:: instance-seg.png
   :figwidth: 50%
   :width: 80%

   Instance image segmentation example
