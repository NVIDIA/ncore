.. Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

Conversion Tools
=================

Data stored in NCore V3 dataformats can be forward converted to the next-generation NCore V4 dataformats using the following tool.

NCore V3 to V4 Conversion
-------------------------

The tool ``//scripts:ncore_3to4`` allows forward conversion of NCore V3 dataformats to NCore V4 dataformats.

Example invocation::

    bazel run //scripts:ncore_3to4 \
        -- \
        --shard-file-pattern=<SHARD_FILE_PATTERN> \
        --output-dir=<TARGET_PATH> \
        [--profile={"default", "separate-sensors", "separate-all"}]

The optional profile flag allows to select different conversion profiles:
- ``default``: Converts all data into a single NCore V4 component group.
- ``separate-sensors``: Converts data into multiple NCore V4 component groups, separating different sensor modalities (e.g. cameras, lidars, radars, etc.).
- ``separate-all``: Converts data into multiple NCore V4 component groups, separating all components individually

Additional group target overwrite flags are available; see the tool's help message for details.
