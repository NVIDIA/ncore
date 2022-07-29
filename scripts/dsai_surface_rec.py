# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os
import logging
import tqdm

import point_cloud_utils as pcu
import numpy as np

from src.py.common.common import load_pc_dat



@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence.', required=True)
@click.option('--output-filename', type=str, help='Name of the ouputfile.', default="reconstructed_mesh")
@click.option('--start-frame', type=click.IntRange(min=0, max_open=True), help='Initial camera frame to be use', default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End camera frame to be used', default=-1)
@click.option('--step-frame', type=click.IntRange(min=1, max_open=True), help='Step used to downsample the number of frames', default=1)
@click.option('--max-dist', type=float, help='Maximum distance for lidar pose.', default=100.0)
@click.option('--n-neighbors', type=int, help='Number of nereast neighbors for normal estimation.', default=20)
@click.option('--trim-distance', type=float, help='Trimming distance', default=0.225)

def run_surface_reconstruction(root_dir, output_filename, max_dist, start_frame, end_frame, step_frame, n_neighbors, trim_distance):
    ''' Given a set of 3D lidar rays in space runs surface reconstruction using SPSR surface reconstruction method (https://hhoppe.com/poissonrecon.pdf)

    Args:
        root_dir (string): path to the root dir containing the dsai converted data
        output_filename (string): filename of the output mesh. By default we use "reconstructed_mesh" which is also the default name in other scripts
        max_dist (float): Ignore fused points greater than this distance from the ego vehicle
        start_frame (int): Start fusing input points at this index
        end_frame (int): Stop fusing input points after this index (-1 = use all frames up to the last)
        step_frame (int): Determines the temporal downsampling rate  (if 1 all frames will be used)
        n_neighbors (int): Number of neighbors used in the k-nn search for the normal estimation
        trim_distance (float): distance used to trimm unwanted parts of the mesh (everything that is further away from the input points will be removed)
    ''' 

    # Initialize the logger
    logger = logging.getLogger(__name__)

    # set up temp paths 
    temp_fused_path = os.path.join(root_dir, 'reconstructed_surface', 'full_pc.ply')
    temp_rec_path = os.path.join(root_dir, 'reconstructed_surface', 'rec_pc.ply')
    
    logger.info("Loading points and ray directions.")

    points, dirs = load_fused_pc(os.path.join(root_dir, 'lidar'),
                            max_dist=max_dist,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            step_frame=step_frame)

    logger.info(f"Loaded {points.shape[0]} points.")

    logger.info("Estimating normal vectors.")
    pid, nf = pcu.estimate_point_cloud_normals_knn(points, n_neighbors, view_directions=dirs,
                                                    drop_angle_threshold=np.pi/2)

    pf = points[pid]

    pcu.save_mesh_vn(temp_fused_path, pf, nf)
    
    # Initialize the logger
    logger.info("Running Poisson surface reconstruction")
    os.system(f"'external/PoissonRecon/PoissonRecon' --in {temp_fused_path} --out {temp_rec_path} " f"--width 0.1 --density --samplesPerNode 1.0 --colors")

    logger.info("Trimming the reconstructed mesh")
    v, f, n, c = pcu.load_mesh_vfnc(temp_rec_path)

    nn_dist, _ = pcu.k_nearest_neighbors(v.astype(np.float32), pf.astype(np.float32), k=2)
    nn_dist = nn_dist[:, 1]
    f_mask = np.stack([nn_dist[f[:, i]] < trim_distance for i in range(f.shape[1])], axis=-1)
    f_mask = np.all(f_mask, axis=-1)

    logger.info("Saving the reconstructed mesh and cleaning up the temp files.")
    output_rec_path =  os.path.join(root_dir, 'reconstructed_surface', f'{output_filename}.ply')
    pcu.save_mesh_vfnc(output_rec_path, v, f[f_mask], n, c)
    # Clean up the temp files
    os.remove(temp_rec_path)
    os.remove(temp_fused_path)


def load_fused_pc(lidar_dir, max_dist=-1, start_frame=0, end_frame=-1, step_frame=1):
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

    # Get all the files and filter them out based on the start, end, step
    files = sorted([os.path.join(lidar_dir, fname) for fname in os.listdir(lidar_dir) if fname.endswith(".dat")])

    all_pts = []
    all_dirs = []
    if end_frame < 0:
        end_frame = len(files) + 1
    files = files[start_frame:end_frame:step_frame]


    for fname in tqdm.tqdm(files):
        pc_data = load_pc_dat(fname)
        points = pc_data[:, 3:6]
        # We take the negative direction such that it point towards the lidar sensor
        dirs = pc_data[:, 0:3] - pc_data[:, 3:6]
        dists = np.linalg.norm(pc_data[:, 3:6] - pc_data[:, 0:3], axis=1)
        
        # Remove dynamic points 
        static_mask = pc_data[:,-1] != 1 # If 1 the point belongs to a dynamic object. Otherwise it can be 0 (static) or -1 (undefined)
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
    run_surface_reconstruction()
