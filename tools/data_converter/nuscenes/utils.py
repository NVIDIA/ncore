# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""nuScenes-specific utilities for the NCore V4 converter."""

from __future__ import annotations

from functools import cache
from typing import Any, Dict, List, Optional

import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


# --- Sensor ID mappings --------------------------------------------------------
# Mapping from NCore sensor ID -> nuScenes channel name

CAMERA_MAP: Dict[str, str] = {
    "camera_front": "CAM_FRONT",
    "camera_front_left": "CAM_FRONT_LEFT",
    "camera_front_right": "CAM_FRONT_RIGHT",
    "camera_back": "CAM_BACK",
    "camera_back_left": "CAM_BACK_LEFT",
    "camera_back_right": "CAM_BACK_RIGHT",
}

LIDAR_ID = "lidar_top"
LIDAR_CHANNEL = "LIDAR_TOP"

RADAR_MAP: Dict[str, str] = {
    "radar_front": "RADAR_FRONT",
    "radar_front_left": "RADAR_FRONT_LEFT",
    "radar_front_right": "RADAR_FRONT_RIGHT",
    "radar_back_left": "RADAR_BACK_LEFT",
    "radar_back_right": "RADAR_BACK_RIGHT",
}

# nuScenes category name -> NCore class_id mapping
NUSCENES_CATEGORY_MAP: Dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.trailer": "trailer",
    "vehicle.emergency.ambulance": "emergency_vehicle",
    "vehicle.emergency.police": "emergency_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}


# --- nuScenes DB helpers -------------------------------------------------------


@cache
def get_nuscenes(version: str, dataroot: str) -> NuScenes:
    """Cached nuScenes DB loader to avoid reloading for multiple scenes."""
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def get_sweep_tokens(nusc: NuScenes, scene_record: Dict[str, Any], channel: str) -> List[str]:
    """Return ordered list of sample_data tokens for all sweeps in a scene for a given channel.

    This includes both keyframe and non-keyframe (interleaved) sweeps.
    """
    result: List[str] = []
    sample_token = scene_record["first_sample_token"]
    sample_record = nusc.get("sample", sample_token)
    sweep_token = sample_record["data"][channel]

    while sweep_token:
        result.append(sweep_token)
        sample_data_record = nusc.get("sample_data", sweep_token)
        sweep_token = sample_data_record["next"]

    return result


def get_sample_records(nusc: NuScenes, scene_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ordered list of keyframe sample records for a scene."""
    result: List[Dict[str, Any]] = []
    sample_token = scene_record["first_sample_token"]

    while sample_token:
        sample_record = nusc.get("sample", sample_token)
        result.append(sample_record)
        sample_token = sample_record["next"]

    return result


def resolve_scene_token(nusc: NuScenes, scene_token: Optional[str], scene_name: Optional[str]) -> Optional[str]:
    """Resolve a scene identifier to its token.

    If scene_token is provided, validate it exists and return it.
    If scene_name is provided, look it up and return its token.
    If neither provided, return None (meaning: convert all scenes).
    """
    if scene_token is not None and scene_name is not None:
        raise ValueError("Specify at most one of --scene-token or --scene-name, not both.")

    if scene_token is not None:
        all_tokens = {s["token"] for s in nusc.scene}
        if scene_token not in all_tokens:
            raise ValueError(
                f"Scene token '{scene_token}' not found in dataset. Available: {sorted(all_tokens)[:5]}..."
            )
        return scene_token

    if scene_name is not None:
        name_to_token = {s["name"]: s["token"] for s in nusc.scene}
        if scene_name not in name_to_token:
            raise ValueError(f"Scene name '{scene_name}' not found. Available: {sorted(name_to_token.keys())[:5]}...")
        return name_to_token[scene_name]

    return None


# --- Cuboid / annotation helpers -----------------------------------------------


def get_boxes_for_sample_data(nusc: NuScenes, sample_data_token: str) -> List[Box]:
    """Get annotation boxes for a sample_data record.

    If the sample_data is a keyframe, returns the annotations for that sample directly.
    If it is an intermediate sweep, interpolates box positions linearly between
    the bracketing keyframes.

    Each returned Box has:
    - .center: [x, y, z] in the global frame
    - .wlh: [width, length, height]
    - .orientation: Quaternion
    - .velocity: [vx, vy, vz] (may contain NaN)
    - .name: category name
    - .token: instance_token (used as track_id)

    Returns:
        List of Box objects in the global (world) coordinate frame.
    """
    sd_record = nusc.get("sample_data", sample_data_token)
    curr_sample_record = nusc.get("sample", sd_record["sample_token"])

    if curr_sample_record["prev"] == "" or sd_record["is_key_frame"]:
        # Keyframe or first sample: return annotations directly
        boxes = []
        for ann_token in curr_sample_record["anns"]:
            record = nusc.get("sample_annotation", ann_token)
            velocity = nusc.box_velocity(record["token"])
            box = Box(
                record["translation"],
                record["size"],
                Quaternion(record["rotation"]),
                velocity=tuple(velocity),
                name=record["category_name"],
                token=record["instance_token"],
            )
            boxes.append(box)
        return boxes

    # Non-keyframe: interpolate between previous and current keyframe annotations
    prev_sample_record = nusc.get("sample", curr_sample_record["prev"])

    curr_ann_recs = [nusc.get("sample_annotation", token) for token in curr_sample_record["anns"]]
    prev_ann_recs = [nusc.get("sample_annotation", token) for token in prev_sample_record["anns"]]

    # Map instance tokens to previous annotation records
    prev_inst_map = {entry["instance_token"]: entry for entry in prev_ann_recs}

    t0 = prev_sample_record["timestamp"]
    t1 = curr_sample_record["timestamp"]
    t = sd_record["timestamp"]

    # Clamp t to [t0, t1] for safety
    t = max(t0, min(t1, t))

    boxes: List[Box] = []
    for curr_ann_rec in curr_ann_recs:
        instance_token = curr_ann_rec["instance_token"]

        if instance_token in prev_inst_map:
            prev_ann_rec = prev_inst_map[instance_token]

            # Interpolate center
            center = [
                np.interp(t, [t0, t1], [c0, c1])
                for c0, c1 in zip(prev_ann_rec["translation"], curr_ann_rec["translation"])
            ]

            # Interpolate orientation via SLERP
            rotation = Quaternion.slerp(
                q0=Quaternion(prev_ann_rec["rotation"]),
                q1=Quaternion(curr_ann_rec["rotation"]),
                amount=(t - t0) / (t1 - t0),
            )
        else:
            # New instance -- use current annotation directly
            center = curr_ann_rec["translation"]
            rotation = Quaternion(curr_ann_rec["rotation"])

        velocity = nusc.box_velocity(curr_ann_rec["token"])
        box = Box(
            center,
            curr_ann_rec["size"],
            rotation,
            velocity=tuple(velocity),
            name=curr_ann_rec["category_name"],
            token=instance_token,
        )
        boxes.append(box)

    return boxes
