# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import json
import os
import logging

from typing import Any, Optional

import click
import numpy as np

from ncore.impl.common.transformations import ecef_2_ENU
from ncore.impl.data.data3 import ShardDataLoader


@click.command()
@click.option(
    "--shard-file-pattern",
    type=str,
    help="Data shard pattern to load (supports range expansion)",
    required=True,
)
@click.option(
    "--ngp-config",
    type=str,
    help="Path to ngp config file with scale and translation (required to output ngp-dependent transformations)",
    required=False,
    default=None,
)
@click.option(
    "--map-ref-lat",
    type=float,
    help="Latitude coordinate of the reference point used for the map ENU coordinate system in degrees (required to output map-dependent transformations)",
    required=False,
    default=None,
)
@click.option(
    "--map-ref-lon",
    type=float,
    help="Longitude coordinate of the reference point used for the map ENU coordinate system in degrees (required to output map-dependent transformations)",
    required=False,
    default=None,
)
@click.option(
    "--map-ref-alt",
    type=float,
    help="Altitude coordinate of the reference point used for the map ENU coordinate system in meters (required to output map-dependent transformations)",
    required=False,
    default=None,
)
@click.option(
    "--output-npz/--no-output-npz",
    default=False,
    help="If enabled, store 'ncore_map_transforms.npz' with transformations next to the ngp-config",
)
@click.option(
    "--output-json/--no-output-json",
    default=True,
    help="If enabled, store 'ncore_map_transforms.json' with transformations next to the ngp-config",
)
@click.option(
    "--output-dir",
    type=str,
    help="Output dir for saving the transform file. If not provided, will save to ngp_config dir instead. It's an error if neither output-dir or ngp-config are provided",
    required=False,
    default=None,
)
def ncore_to_map_transform(
    shard_file_pattern: str,
    ngp_config: Optional[str],
    map_ref_lat: Optional[float],
    map_ref_lon: Optional[float],
    map_ref_alt: Optional[float],
    output_npz: bool,
    output_json: bool,
    output_dir: Optional[str],
):
    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)

    logger.info(f"Shards: {shards}")

    loader = ShardDataLoader(shards)

    # Extract the base pose
    T_ncore_ecef = loader.get_poses().T_rig_world_base

    # Check if the base pose is valid
    # TODO(janickm): explicitly serialize base poses reference frame into shard data to allow for an explicit type-check here
    ncore_ecef_trans_norm = np.linalg.norm(T_ncore_ecef[:, 3])
    if ncore_ecef_trans_norm < 100:  # 100 is just a random number that is small enough to make this suspicious
        logging.warning(
            f"The norm of the NCORE to ECEF translation is suspiciously low ({ncore_ecef_trans_norm} m). Please check that you are using the *global* poses!"
        )

    # Collect mandatory / unconditional output data
    output_data: dict[str, Any] = {
        "T_ncore_ecef": T_ncore_ecef,
        "sequence_id": loader.get_sequence_id(),
    }

    # Collect additional optional output data
    if ngp_config is not None:
        logger.info(f"NGP config: {ngp_config}")
        assert os.path.exists(ngp_config), "Provided NGP config file doesn't exist."

        # Read the NGP config file and extract the scale and offset
        ngp_config_dict = json.load(open(ngp_config))
        scale = ngp_config_dict["scale"]
        offset = ngp_config_dict["offset"]

        # Compute the transformation from the NGP coordinate system to the ECEF
        T_nerf_ngp = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        T_ncore_nerf = np.array(
            [[scale, 0, 0, offset[0]], [0, scale, 0, offset[1]], [0, 0, scale, offset[2]], [0, 0, 0, 1]]
        )
        T_ncore_ngp = T_nerf_ngp @ T_ncore_nerf
        T_ngp_ecef = T_ncore_ecef @ np.linalg.inv(T_ncore_ngp)

        output_data |= {
            "T_ngp_ecef": T_ngp_ecef,
            "ngp_scale": scale,
            "ngp_offset": offset,
        }

        # adapt output if not explicitly set already
        if output_dir is None:
            output_dir = os.path.dirname(ngp_config)
    else:
        logger.warning("Not creating ngp-dependent transformations / data as arguments not provided")

    if all(c is not None for c in [map_ref_lat, map_ref_lon, map_ref_alt]):
        # Compute the transformation from the ECEF coordinate system to the map ENU system
        lat_long_alt = np.array([map_ref_lat, map_ref_lon, map_ref_alt]).reshape(1, 3)
        T_ecef_enu = ecef_2_ENU(lat_long_alt, earth_model="WGS84")

        output_data |= {
            "T_ecef_enu": T_ecef_enu,
            "map_ref_lat_deg": map_ref_lat,
            "map_ref_lon_deg": map_ref_lon,
            "map_ref_alt_m": map_ref_alt,
        }
    else:
        logger.warning("Not creating map-dependent transformations / data as arguments not provided")

    # Print out the transformation matrices
    with np.printoptions(floatmode="unique", linewidth=200, suppress=True):  # print in highest precision

        logger.info(f"T_ncore_ecef:\n{T_ncore_ecef}")

        if (T_ngp_ecef_opt := output_data.get("T_ngp_ecef")) is not None:
            logger.info(f"T_ngp_ecef:\n{T_ngp_ecef_opt}")

        if (T_ecef_enu_opt := output_data.get("T_ecef_enu")) is not None:
            logger.info(f"T_ecef_enu:\n{T_ecef_enu_opt}")
            logger.info(
                f"T_ncore_enu:\n{T_ecef_enu_opt @ T_ncore_ecef}"
            )  # should be used to transform a mesh / "local" world coordinates

        if isinstance(T_ngp_ecef_opt, np.ndarray) and isinstance(T_ecef_enu_opt, np.ndarray):
            logger.info(f"T_ngp_enu:\n{T_ecef_enu_opt @ T_ngp_ecef_opt}")  # should be used to transform a NeRF

    # Save the transformations
    assert output_dir is not None, "Output-dir needs to be specified explicitly if not providing ngp-config"

    if output_npz:
        npz_path = os.path.join(output_dir, "ncore_map_transforms.npz")
        np.savez(npz_path, **output_data)
        logger.info(f"Outputted {npz_path}")
    if output_json:
        json_path = os.path.join(output_dir, "ncore_map_transforms.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    # convert values from numpy to python-internal types if necessary
                    key: value if not isinstance(value, np.ndarray) else value.tolist()
                    for (key, value) in output_data.items()
                },
                f,
                indent=2,
            )
        logger.info(f"Outputted {json_path}")


if __name__ == "__main__":
    ncore_to_map_transform()
