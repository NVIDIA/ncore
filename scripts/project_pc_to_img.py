# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import pickle
import os 
import glob
import random
import numpy as np
import argparse
import time
from PIL import Image
from matplotlib import pyplot as plt 

import sys
sys.path.append('./')
from src.nvidia_utils import (transform_point_cloud, PoseInterpolator, world_points_2_pixel_py, project_camera_rays_2_img)
from src.visualization import plot_points_on_image
from src.common import load_pc_dat, NV_CAMERAS, WAYMO_CAMERAS
from lib import rollingShutterProjection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="Path to the raw data", required=True)
    parser.add_argument("--dataset", type=str, help="Name of the dataset", choices=['nvidia-deepmap', 'waymo', 'nvidia-maglev'], required=True)
    parser.add_argument("--cam_id", type=str, help="Camera ID to be used for projection. If not specified random camera will be used", default= '-1')
    parser.add_argument("--frame_num", type=int, help="Frame number to be used. If not specified random frame will be used", default=-1)
    parser.add_argument("--python", action='store_true', help="If true, rolling shutter projection will also be computed with python code")
    parser.add_argument("--index_digits", type=int, help="The number of integer digits to pad counters in output filenames to", default=6)

    args = parser.parse_args()

    # Select the correct maps
    CAM_IDS = NV_CAMERAS if args.dataset.startswith('nvidia') else WAYMO_CAMERAS
    
    # Check the validity of the input parameters
    if args.cam_id != '-1':
        if args.cam_id not in CAM_IDS:
            print(f'{args.cam_id} is not a valid cam id for dataset {args.dataset}. A random cam will be selected instead.')
            cam_id =  random.choice(CAM_IDS)
        else:
            cam_id = args.cam_id
    else:
        print('Camera id not provided. A random camera will be selected.')
        cam_id =  random.choice(CAM_IDS)


    # Get all the frames of the selected camera
    img_list = sorted(glob.glob(os.path.join(args.root_dir, f'images/image_{cam_id}/*')))
    img_list = [filename for filename in img_list if os.path.splitext(filename)[1][1:].strip().lower() == 'jpeg' and len(os.path.splitext(filename.split(os.sep)[-1])[0]) == args.index_digits]

    if args.frame_num != -1:
        if os.path.join(os.path.join(args.root_dir, f'images/image_{cam_id}/{str(args.frame_num).zfill(args.index_digits)}.jpeg')) in img_list:
            rgb_frame_path = os.path.join(os.path.join(args.root_dir, f'images/image_{cam_id}/{str(args.frame_num).zfill(args.index_digits)}.jpeg'))
        else:
            print(f'Frame {args.frame_num} is not available for cam {cam_id} in {args.dataset} dataset. A random frame will be selected instead.')
            rgb_frame_path = random.choice(img_list)
    else:
        print('Frame number not provided. A random frame will be selected.')
        rgb_frame_path = random.choice(img_list)


	# Load image
    img = np.array(Image.open(rgb_frame_path).convert('RGB'))

	# Load metadata
    with open(os.path.join(rgb_frame_path.replace('.jpeg','.pkl')), 'rb') as f:
        metadata = pickle.load(f)

    # Load point cloud 
    # Find the closest lidar frame based on the timestamp
    t_sof = metadata['ego_pose_timestamps'][0]
    lidar_timestamps = np.load(os.path.join(args.root_dir, 'lidar/timestamps.npz'))['timestamps']
    lidar_frame_idx = np.argmin(np.abs(lidar_timestamps - t_sof))

    # Load the PC dat file and extract the end point coordinates
    pc_data = load_pc_dat(os.path.join(args.root_dir, 'lidar', f'{str(lidar_frame_idx).zfill(args.index_digits)}.dat'))
    pc = pc_data[:,3:6]

    # Check if the ground truth projections are available
    gt_flag = False
    if os.path.exists(os.path.join(args.root_dir, 'lidar', f'{str(lidar_frame_idx).zfill(args.index_digits)}_cp.npz')):
        gt_projections = np.load(os.path.join(args.root_dir, 'lidar', f'{str(lidar_frame_idx).zfill(args.index_digits)}_cp.npz'))['cp']
        gt_flag = True

	# Project the points without considering rolling shutter
    pose_timestamps = metadata['ego_pose_timestamps']
    poses = np.stack((metadata['ego_pose_s'], metadata['ego_pose_e']))
    t_mof = 0.5 * np.sum(pose_timestamps) 
    pose_interpolator = PoseInterpolator(poses, pose_timestamps)

    cam_pose_global = pose_interpolator.interpolate_to_timestamps(t_mof)

    single_pose = np.linalg.inv(metadata['T_cam_rig']) @ np.linalg.inv(cam_pose_global[0]) 
    pc_cam = transform_point_cloud(pc, single_pose)

    pixel_coords, valid_idx = project_camera_rays_2_img(pc_cam, metadata)
    pixel_coords = pixel_coords[valid_idx,:]
    pc_cam = pc_cam[valid_idx,:]

    dist = np.linalg.norm(pc_cam,axis=1, keepdims=True)

	# Project the points by considering rolling shutter
    start_time = time.time()
    pixel_coords_rs, trans_matrices_rs, valid_idx_rs = rollingShutterProjection(pc, metadata, iter=10)
    print(f"C++ imp requires: {time.time() - start_time:0.3f} s for the rolling shutter projection of {pc.shape[0]} points")

    # Compute the distance to the points in the camera coordinate system
    transformed_points = (trans_matrices_rs[:,:3,:3] @ pc[valid_idx_rs,:,None] + trans_matrices_rs[:,:3,3:4]).squeeze(-1)
    dist_rs = np.linalg.norm(transformed_points,axis=1,keepdims=True)

    # If gt_flag exists compute some statistics
    if gt_flag:
        assert gt_projections.shape[0] == pc.shape[0], "The number of lidar points doesn't match the number of GT projections"
        idx_first = np.where(gt_projections[:,0] == int(cam_id) + 1)[0]
        idx_second = np.where(gt_projections[:,3] == int(cam_id) + 1)[0] 
        all_gt_idx = np.concatenate((idx_first,idx_second))
        gt_pixel_values = np.concatenate((gt_projections[idx_first,1:3], gt_projections[idx_second,4:6]), axis=0)
        gt_dist = np.linalg.norm(pc_data[all_gt_idx,3:6] - pc_data[all_gt_idx,0:3], axis=1)

        diff_gt_rs = []
        for valid_proj, proj_idx in zip(pixel_coords_rs, valid_idx_rs):
            corr_gt_indx = np.where( all_gt_idx == proj_idx)[0]
            if corr_gt_indx.shape[0] > 1:
                corr_gt_indx = corr_gt_indx[0]

            if corr_gt_indx.size > 0:
                diff_gt_rs.append(np.linalg.norm(gt_pixel_values[corr_gt_indx] - valid_proj))

        print("Difference between GT lidar projections and our Rolling shutter implementation")
        print(f"Mean: {np.mean(diff_gt_rs):0.3f} px | Max: {np.max(diff_gt_rs):0.3f} px | Min: {np.min(diff_gt_rs):0.3f} px | Std. {np.std(diff_gt_rs):0.3f} px |")

        diff_gt_naive_= []
        for valid_proj, proj_idx in zip(pixel_coords, valid_idx):
            corr_gt_indx = np.where( all_gt_idx == proj_idx)[0]
            if corr_gt_indx.shape[0] > 1:
                corr_gt_indx = corr_gt_indx[0]

            if corr_gt_indx.size > 0:
                diff_gt_naive_.append(np.linalg.norm(gt_pixel_values[corr_gt_indx] - valid_proj))

        print("Difference between GT lidar projections and naive projection")
        print(f"Mean: {np.mean(diff_gt_naive_):0.3f} px | Max: {np.max(diff_gt_naive_):0.3f} px | Min: {np.min(diff_gt_naive_):0.3f} px | Std. {np.std(diff_gt_naive_):0.3f} px |")


    # Compute also the python version (without iterative process)
    if args.python:
        start_time = time.time()
        pixel_coords_rs_py, trans_matrices_rs_py, valid_idx_rs_py = world_points_2_pixel_py(pc, metadata)
        # Compute the distance to the points in the camera coordinate system
        transformed_points = (trans_matrices_rs_py[:,:3,:3] @ pc[valid_idx_rs_py,:,None] + trans_matrices_rs_py[:,:3,3:4]).squeeze(-1)
        dist_rs_py = np.linalg.norm(transformed_points,axis=1,keepdims=True)

        print(f"Python imp requires: {time.time() - start_time:0.3f} s for the rolling shutter projection of {pc.shape[0]} points")


    plot_points_on_image(np.concatenate((pixel_coords[:,:2], dist),axis=1), img, "Projection without considering rolling shutter", point_size=6.0)
    plot_points_on_image(np.concatenate((pixel_coords_rs[:,:2], dist_rs),axis=1), img, "Projection with rolling shutter (c++ implementation)", point_size=6.0)

    if gt_flag:
        plot_points_on_image(np.concatenate((gt_pixel_values[:,:2], gt_dist[:,None]),axis=1), img, "GT Projections", point_size=6.0)

    if args.python:
        plot_points_on_image(np.concatenate((pixel_coords_rs_py[:,:2], dist_rs_py),axis=1), img, "Projection with rolling shutter (python implementation)", point_size=6.0)

    plt.show()
