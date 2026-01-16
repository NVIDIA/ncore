# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Union

from . import types


JsonLike = Union[
    Dict[str, "JsonLike"],
    List["JsonLike"],
    str,
    int,
    float,
    bool,
    None,
    # special-case shouldn't be needed, but required to make mypy happy
    List[int],
]


def evaluate_file_pattern(pattern: str, skip_suffixes: Iterable[str] = ()) -> List[str]:
    """Given a file-pattern returns a list of matching and existing files

    Supported patterns (mutually exclusive):
    - integer-ranges: '/some/path/file-[1-3]' will be expanded to [/some/path/file-1, /some/path/file-2, /some/path/file-3]

    """

    pattern_basepath = Path(pattern).parent
    pattern_name = Path(pattern).name

    evaluated_name_patterns = []

    # expand integer ranges like '[1-13]'
    if range_match := re.search(r"\[(\d+)-(\d+)\]", pattern_name):
        low = int(range_match.group(1))
        high = int(range_match.group(2))

        for i in range(low, high + 1):
            evaluated_name_patterns.append(pattern_name.replace(f"[{low}-{high}]", str(i) + "-"))
    else:
        evaluated_name_patterns.append(pattern_name)

    matches: set[Path] = set()
    for evaluated_pattern in evaluated_name_patterns:
        for candidate in pattern_basepath.iterdir():
            if candidate.name.startswith(evaluated_pattern):
                skip = False
                for skip_suffix in skip_suffixes:
                    if str(candidate).endswith(skip_suffix):
                        skip = True
                        break
                if not skip:
                    matches.add(candidate)

    return [str(match) for match in list(matches)]


def encode_camera_model_parameters(camera_model_parameters: types.ConcreteCameraModelParametersUnion) -> Dict:
    """Encodes camera intrinsic model parameters to serializable model-typed dictionary"""

    encoded = {
        "camera_model_type": camera_model_parameters.type(),
        "camera_model_parameters": camera_model_parameters.to_dict(),
    }

    # Store type of external distortion, if available
    if camera_model_parameters.external_distortion_parameters:
        encoded["external_distortion_type"] = camera_model_parameters.external_distortion_parameters.type()

    return encoded


def decode_camera_model_parameters(encoded_parameters: Mapping) -> types.ConcreteCameraModelParametersUnion:
    """Decodes model-typed dictionary parameters specific to the camera's intrinsic model"""

    camera_model_type = encoded_parameters["camera_model_type"]

    # Copy as we might modify the dictionary in place
    camera_model_parameters = encoded_parameters["camera_model_parameters"].copy()

    # Hook up typed external distortion type, if present
    external_distortion_type: Optional[str] = encoded_parameters.get("external_distortion_type")
    if external_distortion_type is not None:
        if external_distortion_type == "bivariate-windshield":
            camera_model_parameters["external_distortion_parameters"] = (
                types.BivariateWindshieldModelParameters.from_dict(
                    camera_model_parameters["external_distortion_parameters"]
                )
            )
        else:
            raise ValueError(f"Unknown external distortion type: {external_distortion_type}")

    # Return typed camera model parameters
    if camera_model_type == "ftheta":
        return types.FThetaCameraModelParameters.from_dict(camera_model_parameters)
    elif camera_model_type in [
        "opencv-pinhole",
        # keep 'pinhole' for backwards-compatibility with existing data
        "pinhole",
    ]:
        return types.OpenCVPinholeCameraModelParameters.from_dict(camera_model_parameters)
    elif camera_model_type == "opencv-fisheye":
        return types.OpenCVFisheyeCameraModelParameters.from_dict(camera_model_parameters)

    raise ValueError(f"Unknown camera model type: {camera_model_type}")


def encode_lidar_model_parameters(lidar_model_parameters: types.ConcreteLidarModelParametersUnion) -> Dict:
    """Encodes lidar intrinsic model parameters to serializable model-typed dictionary"""

    encoded = {
        "lidar_model_type": lidar_model_parameters.type(),
        "lidar_model_parameters": lidar_model_parameters.to_dict(),
    }

    return encoded


def decode_lidar_model_parameters(encoded_parameters: Mapping) -> types.ConcreteLidarModelParametersUnion:
    """Decodes model-typed dictionary parameters specific to the lidars's intrinsic model"""

    lidar_model_type = encoded_parameters["lidar_model_type"]

    # Return typed lidar model parameters
    if lidar_model_type == types.RowOffsetStructuredSpinningLidarModelParameters.type():
        return types.RowOffsetStructuredSpinningLidarModelParameters.from_dict(
            encoded_parameters["lidar_model_parameters"]
        )

    raise ValueError(f"Unknown lidar model type: {lidar_model_type}")
