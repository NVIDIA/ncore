# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import json
import os
import logging

import click
import numpy as np

from dsai.impl.common.transformations import ecef_2_ENU
from dsai.impl.data.data3 import ShardDataLoader


@click.command()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
@click.option('--ngp-config', type=str, help='Path to ngp config file with scale and translation', required=True)
@click.option('--map-ref-lat',
              type=float,
              help='Latitude coordinate of the reference point used for the map ENU coordinate system in degrees!',
              required=True)
@click.option('--map-ref-lon',
              type=float,
              help='Longitude coordinate of the reference point used for the map ENU coordinate system in degrees!',
              required=True)
@click.option('--map-ref-alt',
              type=float,
              help='Altitude coordinate of the reference point used for the map ENU coordinate system in meters!',
              required=True)
def dsai_to_map_transform(shard_file_pattern: str, ngp_config: str, map_ref_lat: float, map_ref_lon: float,
                          map_ref_alt: float):

    assert os.path.exists(ngp_config), "Provided NGP config file doesn't exist."

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)

    logger.info(f"Shards: {shards}")
    logger.info(f"NGP config: {ngp_config}")

    loader = ShardDataLoader(shards)

    # Extract the base pose
    T_dsai_ecef = loader.get_poses().T_rig_world_base

    # Check if the base pose is valid
    dsai_ecef_trans_norm = np.linalg.norm(T_dsai_ecef[:, 3])
    if dsai_ecef_trans_norm < 100:  # 100 is just a random number that is small enough to make this suspicious
        logging.warning(
            f"The norm of the DSAI to ECEF translation is suspiciously low ({dsai_ecef_trans_norm} m). Please check that you are using the global poses!"
        )

    # Read the NGP config file and extract the scale and offset
    ngp_config_dict = json.load(open(ngp_config))
    scale = ngp_config_dict['scale']
    offset = ngp_config_dict['offset']

    # Compute the transformation from the NGP coordinate system to the ECEF
    T_nerf_ngp = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    T_dsai_nerf = np.array([[scale, 0, 0, offset[0]], [0, scale, 0, offset[1]], [0, 0, scale, offset[2]], [0, 0, 0, 1]])
    T_dsai_ngp = T_nerf_ngp @ T_dsai_nerf
    T_ngp_ecef = T_dsai_ecef @ np.linalg.inv(T_dsai_ngp)

    # Compute the transformation from the ECEF coordiante system to the map ENU system
    lat_long_alt = np.array([map_ref_lat, map_ref_lon, map_ref_alt]).reshape(1, 3)
    T_ecef_enu = ecef_2_ENU(lat_long_alt, earth_model='WGS84')

    # Print out the transformation matrices
    with np.printoptions(floatmode='unique', linewidth=200, suppress=True):  # print in highest precision
        logger.info(f"T_ngp_ecef:\n{T_ngp_ecef}")
        logger.info(f"T_ecef_enu:\n{T_ecef_enu}")
        logger.info(f"T_dsai_ecef:\n{T_dsai_ecef}")
        logger.info(f"T_ngp_enu:\n{T_ecef_enu @ T_ngp_ecef}")  # should be used to transform a NeRF
        logger.info(f"T_dsai_enu:\n{T_ecef_enu @ T_dsai_ecef}"
                    )  # should be used to transform a mesh / "local" world coordinates

    # Save the transformations
    ngp_config_dir = os.path.dirname(ngp_config)
    dsai_ds_path = os.path.join(ngp_config_dir, 'T_dsai_ds.npz')
    np.savez(dsai_ds_path, T_ngp_ecef=T_ngp_ecef, T_ecef_enu=T_ecef_enu, T_dsai_ecef=T_dsai_ecef)
    logger.info(f'outputted {dsai_ds_path}')


if __name__ == '__main__':
    dsai_to_map_transform()
