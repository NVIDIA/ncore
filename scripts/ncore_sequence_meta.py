# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import hashlib
import json
import logging

from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import click

from ncore.impl.data.data3 import ShardDataLoader


def md5(path: Path, chunk_size: int = 128 * 2**9) -> str:
    """Compute the MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@click.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option(
    "--shard-file-skip-suffix",
    "shard_file_skip_suffixes",
    multiple=True,
    type=str,
    help="Suffixes to skip when evaluating shard file pattern",
    default=None,
)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--output-file",
    type=str,
    default=None,
    help="Filename of generated file (json) - <sequence_id>.json will be used by default if not provided",
    required=False,
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option("--debug", is_flag=True, default=False, help="Enables debug logging outputs")
def ncore_sequence_meta(
    shard_file_pattern: str,
    shard_file_skip_suffixes: Tuple[str],
    output_dir: str,
    output_file: Optional[str],
    open_consolidated: bool,
    debug: bool,
):
    """Summarizes and exports data-ranges within a virtual shard sequence"""

    # Initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
    )
    logger = logging.getLogger(__name__)

    loader = ShardDataLoader(
        ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern, skip_suffixes=shard_file_skip_suffixes),
        open_consolidated=open_consolidated,
    )

    ## Sequence-wide information
    output: dict[str, object] = {
        "sequence_id": loader.get_sequence_id(),
        "pose-range": {
            "start-timestamp_us": int((sequence_pose_timestamps_us := loader.get_poses().T_rig_world_timestamps_us)[0]),
            "end-timestamp_us": int(sequence_pose_timestamps_us[-1]),
            "num-poses": len(sequence_pose_timestamps_us),
        },
        "shard-ids": loader.get_shard_ids(),
    }
    sequence_start_timestamp_us, sequence_end_timestamp_us = (
        sequence_pose_timestamps_us[0],
        sequence_pose_timestamps_us[-1],
    )

    ## Shard-wide information
    shards = []
    shard_pose_offset = 0
    sensor_frame_offset: dict[str, int] = defaultdict(int)
    for shard_idx, (shard_id, shard_file) in enumerate(zip(loader.get_shard_ids(), loader.get_shard_paths())):
        start_shard_idx = shard_idx
        stop_shard_idx = shard_idx + 1

        shard_pose_timestamps_us = loader.get_poses(
            start_shard_idx=start_shard_idx, stop_shard_idx=stop_shard_idx
        ).T_rig_world_timestamps_us

        # sanity checks
        assert (
            sequence_start_timestamp_us <= shard_pose_timestamps_us[0]
            and shard_pose_timestamps_us[-1] <= sequence_end_timestamp_us
        ), f"shard {shard_idx} pose timestamps inconsistent with sequence pose timestamps"

        shard = {
            "id": shard_id,
            "path": (shard_path := Path(shard_file)).name,
            "md5": md5(shard_path),
            "pose-range": {
                "start-timestamp_us": int(shard_pose_timestamps_us[0]),
                "end-timestamp_us": int(shard_pose_timestamps_us[-1]),
                "num-poses": len(shard_pose_timestamps_us),
                "sequence-pose-offset": shard_pose_offset,
                "sequence-time-offset_us": int(shard_pose_timestamps_us[0] - sequence_start_timestamp_us),
                "sequence-time-offset_sec": (shard_pose_timestamps_us[0] - sequence_start_timestamp_us) / 1e6,
            },
        }
        shard_pose_offset += len(shard_pose_timestamps_us)

        sensors = {}

        for sensor_id in loader.get_sensor_ids():
            sensor = loader.get_sensor(sensor_id)

            sensor_frame_timestamps_us = sensor.get_frames_timestamps_us(
                start_shard_idx=start_shard_idx, stop_shard_idx=stop_shard_idx
            )

            # sanity checks
            assert (
                sequence_start_timestamp_us <= sensor_frame_timestamps_us[0]
                and sensor_frame_timestamps_us[-1] <= sequence_end_timestamp_us
            ), f"sensor {sensor_id} frame timestamps inconsistent with sequence pose timestamps"

            sensors[sensor_id] = {
                "frame-range": {
                    "start-timestamp_us": int(sensor_frame_timestamps_us[0]),
                    "end-timestamp_us": int(sensor_frame_timestamps_us[-1]),
                    "num-frames": len(sensor_frame_timestamps_us),
                    "sequence-frame-offset": sensor_frame_offset[sensor_id],
                    "sequence-time-offset_us": int(sensor_frame_timestamps_us[0] - sequence_start_timestamp_us),
                    "sequence-time-offset_sec": (sensor_frame_timestamps_us[0] - sequence_start_timestamp_us) / 1e6,
                },
            }
            sensor_frame_offset[sensor_id] += len(sensor_frame_timestamps_us)

        shard["sensors"] = sensors

        shards.append(shard)

    output["shards"] = shards

    ## Serialize output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_file:
        output_path /= output_file
    else:
        output_path /= f"{loader.get_sequence_id()}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote meta data {str(output_path)}")


if __name__ == "__main__":
    ncore_sequence_meta()
