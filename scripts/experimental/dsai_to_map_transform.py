# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import json
import os 
import logging
import numpy as np

from src.py.common.transformations import ecef_2_ENU

@click.command()
@click.option('--root-dir', type=str, help='Path to processed data', required=True)
@click.option('--ngp-config', type=str, help='Path to ngp config file with scale and translation', required=True)
@click.option('--map-ref-lat', type=float, help='Latitude coordinate of the reference point used for the map ENU coordinate system', required=True)
@click.option('--map-ref-lon', type=float, help='Longitude coordinate of the reference point used for the map ENU coordinate system', required=True)
@click.option('--map-ref-alt', type=float, help='Altitude coordinate of the reference point used for the map ENU coordinate system', required=True)

def dsai_to_map_transform(root_dir: str, ngp_config: str, map_ref_lat: float, map_ref_lon: float, map_ref_alt: float):

    assert os.path.exists(ngp_config), "Procided NGP config file doesn't exist."
    assert os.path.exists(root_dir), "Procided root_dir path doesn't exist."

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info(f"Root dir: {root_dir}")
    logger.info(f"NGP config: {ngp_config}")

    # Extract the base pose 
    base_pose = np.load(os.path.join(root_dir, 'poses/poses.npz'))['base_pose']

    # Read the NGP config file and extract the scale and offset
    ngp_config_dict = json.load(open(ngp_config))
    scale = ngp_config_dict['scale']
    offset = ngp_config_dict['offset']
    experiment_name = ngp_config.split(os.sep)[-2]

    # Compute the transformation from the NGP coordinate system to the ECEF
    T_nerf_ngp = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]])
    T_dsai_nerf = np.array([[scale,0,0,offset[0]],[0,scale,0,offset[1]],[0,0,scale,offset[2]],[0,0,0,1]])
    T_dsai_ngp = T_nerf_ngp @ T_dsai_nerf
    T_ngp_ecef = base_pose @ np.linalg.inv(T_dsai_ngp)

    np.set_printoptions(suppress=True)
    logger.info(f"T_ngp_ecef:\n {T_ngp_ecef}")

    # Compute the transformation from the ECEF coordiante system to the map ENU system
    lat_long_alt = np.array([map_ref_lat, map_ref_lon, map_ref_alt]).reshape(1,3)
    T_ecef_enu = ecef_2_ENU(lat_long_alt, earth_model='WGS84')


    logger.info(f"T_ecef_enu:\n {T_ecef_enu}")
    logger.info(f"base_pose:\n {base_pose}")


    logger.info(f"T_ngp_enu:\n {T_ecef_enu @ T_ngp_ecef}")


    # Save the transformations
    os.makedirs(os.path.join(root_dir, 'poses', experiment_name), exist_ok=True)
    np.savez(os.path.join(root_dir, 'poses', experiment_name, 'T_dsai_ds.npz'), T_ngp_ecef=T_ngp_ecef, 
                                                              T_ecef_enu=T_ecef_enu,
                                                              base_pose=base_pose)

if __name__ == '__main__':
    dsai_to_map_transform()