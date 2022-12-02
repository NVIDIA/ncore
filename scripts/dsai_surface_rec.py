# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging
import tqdm
import tempfile
import os

from typing import Optional
from pathlib import Path

import click
import numpy as np
import point_cloud_utils as pcu

from src.dsai_internal.common.transformations import transform_point_cloud
from src.dsai_internal.data.data2 import DataLoader, PointCloudSensor


@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence', required=True)
@click.option('--sensor-id', type=str, help='Sensor to construct surface from', default='lidar_gt_top_p128_v4p5')
@click.option(
    '--output-dir',
    type=str,
    help='Path to the output folder (will output into \'source-folder/reconstructed_surface\' if not provided)',
    default=None)
@click.option('--output-filename', type=str, help='Name of the output file', default="reconstructed_mesh")
@click.option('--start-frame',
              type=click.IntRange(min=0, max_open=True),
              help='Initial frame to be use',
              default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End frame to be used', default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)
@click.option('--max-dist',
              type=float,
              help='Ignore fused points greater than this distance from the ego vehicle',
              default=100.0)
@click.option('--n-neighbors',
              type=int,
              help='Number of neighbors used in the k-nn search for the normal estimation',
              default=20)
@click.option(
    '--trim-distance',
    type=float,
    help=
    'Trimming distance to trimm unwanted parts of the mesh (everything that is further away from the input points will be removed)',
    default=0.225)
def dsai_surface_rec(root_dir: str, sensor_id: str, output_dir: Optional[str], output_filename: str, max_dist: float,
                     start_frame: int, end_frame: int, step_frame: int, n_neighbors: int, trim_distance: float):
    ''' Given a set of 3D lidar rays in space runs surface reconstruction using SPSR surface reconstruction method (https://hhoppe.com/poissonrecon.pdf) '''

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # set up output paths
    if not output_dir:
        output_path = Path(root_dir) / 'reconstructed_surface'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading points and ray directions.")

    points, dirs = load_fused_pc(root_dir,
                                 sensor_id,
                                 max_dist=max_dist,
                                 start_frame=start_frame,
                                 end_frame=end_frame,
                                 step_frame=step_frame)

    logger.info(f"Loaded {points.shape[0]} points.")

    logger.info("Estimating normal vectors.")
    pid, nf = pcu.estimate_point_cloud_normals_knn(points,
                                                   n_neighbors,
                                                   view_directions=dirs,
                                                   drop_angle_threshold=np.pi / 2)

    pf = points[pid]

    with tempfile.NamedTemporaryFile(suffix='.ply') as temp_fp, tempfile.NamedTemporaryFile(suffix='.ply') as temp_rp:
        pcu.save_mesh_vn(temp_fp.name, pf, nf)

        logger.info("Running Poisson surface reconstruction")
        os.system(f"'external/PoissonRecon/PoissonRecon' --in {temp_fp.name} --out {temp_rp.name} "
                  f"--width 0.1 --density --samplesPerNode 1.0 --colors")

        logger.info("Trimming the reconstructed mesh")
        v, f, n, c = pcu.load_mesh_vfnc(temp_rp.name)

    nn_dist, _ = pcu.k_nearest_neighbors(v.astype(np.float32), pf.astype(np.float32), k=2)
    nn_dist = nn_dist[:, 1]
    f_mask = np.stack([nn_dist[f[:, i]] < trim_distance for i in range(f.shape[1])], axis=-1)
    f_mask = np.all(f_mask, axis=-1)

    logger.info("Saving the reconstructed mesh and cleaning up the temp files.")
    pcu.save_mesh_vfnc(str(output_path / f'{output_filename}.ply'), v, f[f_mask], n, c)


def load_fused_pc(root_dir, sensor_id, max_dist=-1, start_frame=0, end_frame=-1, step_frame=1):
    ''' Load the individual point cloud spins and accumulates them to a single point cloud

    Args:
        lidar_dir (string): path to the lidar data
        max_dist (float): Ignore fused points greater than this distance from the ego vehicle
        start_frame (int): Start fusing input points at this index
        end_frame (int): Stop fusing input points after this index (-1 = use all frames up to the last)
        step_frame (int): Determines the temporal downsampling rate  (if 1 all frames will be used)
        n_neighbors (int): Number of neighbors used in the k-nn search for the normal estimation
        trim_distance (float): distance used to trimm unwanted parts of the mesh (everything that is further away from the input points will be removed)
    '''

    # Load point-cloud data
    loader = DataLoader(root_dir)
    sensor = loader.get_sensor(sensor_id)
    assert isinstance(sensor, PointCloudSensor), 'only point-cloud sensors supported'

    all_pts = []
    all_dirs = []

    for frame_index in tqdm.tqdm(sensor.get_frame_index_range(start_frame, end_frame, step_frame)):
        T_sensor_to_world = sensor.get_frame_T_sensor_world(frame_index)
        points = transform_point_cloud(sensor.get_frame_data(frame_index, 'xyz_e'), T_sensor_to_world)
        dirs = transform_point_cloud(sensor.get_frame_data(frame_index, 'xyz_s'), T_sensor_to_world) - points
        dists = np.linalg.norm(dirs, axis=1)

        # Remove dynamic points
        static_mask = sensor.get_frame_data(
            frame_index, 'dynamic_flag'
        ) != 1  # If 1 the point belongs to a dynamic object. Otherwise it can be 0 (static) or -1 (undefined)
        points, dirs, dists = points[static_mask], dirs[static_mask], dists[static_mask]

        # Filter the points based on the maximum distance
        if max_dist > 0:
            dist_mask = dists < max_dist
            points, dirs = points[dist_mask], dirs[dist_mask]

        all_pts.append(points)
        all_dirs.append(dirs)

    all_pts = np.concatenate(all_pts)
    all_dirs = np.concatenate(all_dirs)
    all_dirs /= np.linalg.norm(all_dirs, axis=-1, keepdims=True)

    return all_pts, all_dirs


if __name__ == "__main__":
    dsai_surface_rec()
