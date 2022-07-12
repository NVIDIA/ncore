#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import sys
sys.path.append('./')

                                                                                                                                 
from src.common import NV_CAMERAS, WAYMO_CAMERAS, R_NVIDIA_NGP, R_WAYMO_NGP, RS_DIR_TO_NGP, average_camera_pose
import numpy as np
from PIL import Image
# PIL pollutes the CL output with debug messages
# TODO: check for better solution
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
import json
import math
import os
import pickle
import copy

@click.group()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence.', required=True)
@click.option('--experiment-name', type=str, help='Name of the experiment.', required=True)
@click.option('--start-frame', type=click.IntRange(min=0, max_open=True), help='Initial camera frame to be use', required=True)
@click.option('--end-frame', type=click.IntRange(min=1, max_open=True), help='End camera frame to be used', required=True)
@click.option('--step-frame', type=click.IntRange(min=1, max_open=True), help='Step used to downsample the number of frames', default=1)
@click.option('--cameras', '--c', multiple=True, type=int, help='Cameras to be used (Multiple value option.)', default=[1])    
@click.option('--max-dist', type=float, help='Maximum distance from each camera pose.', default=50.0)
@click.option('--use-lidar', is_flag=True, default=False, help="Use also lidar point clouds")
@click.option("--index_digits", type=int, help="The number of integer digits to pad counters in output filenames to", default=6)
@click.version_option('0.1')
@click.pass_context
def cli(ctx, *_, **kwargs):
    ctx.obj = kwargs

@cli.command()
@click.pass_context
def waymo(ctx, *_, **kwargs):
    
    # Extract the CL arguments 
    root_dir = ctx.obj['root_dir']
    exp_name = ctx.obj['experiment_name']
    start_frame = ctx.obj['start_frame']
    end_frame = ctx.obj['end_frame']
    step_frame = ctx.obj['step_frame']
    cameras = ctx.obj['cameras']
    index_digits = ctx.obj['index_digits']
    max_dist = ctx.obj['max_dist']
    use_lidar = ctx.obj['use_lidar']

    # Check that the input arguments are valid
    assert start_frame < end_frame, "End frame index is smaller that start frame one."
    assert not os.path.exists(os.path.join(root_dir, f'ngp_configs/{exp_name}')), "Experiment with the same name already exists."
    for cam in cameras:
        assert str(cam).zfill(2) in WAYMO_CAMERAS, "Invalid camera selected for Nvidia dataset."
    
    # Create the output path 
    output_dir = os.path.join(root_dir, f'ngp_configs/{exp_name}') 
    os.makedirs(output_dir)
    
    camera_data = {}
    for cam in cameras: 
        camera_data[cam] = {}
        
        with open(os.path.join(root_dir, f'images/image_{str(cam).zfill(2)}', f'{str(start_frame).zfill(6)}.pkl'),'rb') as file:
            camera_metadata = pickle.load(file)

        # Get the focal length, image width and height
        focal_length_x = camera_metadata['intrinsic'][0]
        focal_length_y = camera_metadata['intrinsic'][1]
        w, h = camera_metadata['img_width'],  camera_metadata['img_height']
        
        # Compute the angular field of view
        fov_angle_x = math.atan(w/(focal_length_x*2))*2
        fov_angle_y = math.atan(h/(focal_length_y*2))*2
        camera_data[cam]['angle_x'] = fov_angle_x
        camera_data[cam]['angle_y'] = fov_angle_y
        camera_data[cam]['w'] = w
        camera_data[cam]['h'] = h

        # Set the UP vector (only used for visualization purposes withing NGP)
        camera_data[cam]['up'] = np.array([0, 0, 1])

        # Load all the images and the corresponding metadata (to average the pose we neglect the selected step size as all images will be used for testing)
        all_img = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'{str(idx).zfill(index_digits)}.jpeg') for idx in range(start_frame, end_frame)]
        all_img_mask = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'mask_{str(idx).zfill(index_digits)}.png') for idx in range(start_frame, end_frame)]
        all_metadata = [img.replace('.jpeg','.pkl') for img in all_img ]

        # Resave all image masks 
        for img_mask_path in all_img_mask:
            img = Image.open(img_mask_path)
            img.save(img_mask_path.replace('mask_', 'dynamic_mask_'), bits=1,optimize=True)

        # Iterate over the poses 
        T_cam_rig = []
        poses_start = []
        poses_end = []

        # Iterate over all the frames and save their metadata
        for cam_pose in all_metadata:
            with open(cam_pose,'rb') as file:
                camera_metadata = pickle.load(file)

            T_cam_rig.append(camera_metadata['T_cam_rig'])
            
            T_cam_world_start = camera_metadata['ego_pose_s'] @ np.array(T_cam_rig)[0]
            T_cam_world_start[:3,:3] = T_cam_world_start[:3,:3] @ R_WAYMO_NGP
            poses_start.append(T_cam_world_start)

            T_cam_world_end = camera_metadata['ego_pose_e'] @ np.array(T_cam_rig)[0]
            T_cam_world_end[:3,:3] = T_cam_world_end[:3,:3] @ R_WAYMO_NGP
            poses_end.append(T_cam_world_end)
                   
        poses_start = np.stack(poses_start)
        poses_end = np.stack(poses_end)

        # Concatenate the bottom row
        camera_data[cam]['poses_start'] = poses_start
        camera_data[cam]['poses_end'] = poses_end
        camera_data[cam]['intrinsic'] =  camera_metadata['intrinsic']

        # Extract the rolling shutter parameters such y = a + b*x + c*y (Waymo rolling shutter is column wise) 
        a, b, c = RS_DIR_TO_NGP[camera_metadata['rolling_shutter_direction']]            
        camera_data[cam]['rolling_shutter'] = np.array([a, b, c])

    # Combine all the poses and compute the scaling factor and centroid, use the start timestamp pose as approximation
    all_poses = []
    for cam in cameras: 
        all_poses.append(camera_data[cam]['poses_start'])
    all_poses = np.concatenate(all_poses,axis=0)

    pose_avg, extent = average_camera_pose(all_poses)
    scale_factor = 1/ ((extent/2 + max_dist) / 8.0) 
    offset = -(pose_avg * scale_factor) + np.array([0.5,0.5,0.5]) # Instant NGP assumes that the scenes are centered at 0.5^3 not at 0!
        
    # Generate a config file for each of the cameras
    for cam_idx, cam in enumerate(cameras): 

        out_train={"aabb_scale":16, 
                "n_extra_learnable_dims" : 8, 
                "camera_angle_x":camera_data[cam]['angle_x'],
                "camera_angle_y":camera_data[cam]['angle_y'], 
                "up":camera_data[cam]['up'].tolist(), 
                "w": float(w), 
                "h": float(h), 
                "cx": float(camera_data[cam]['intrinsic'][2]), 
                "cy": float(camera_data[cam]['intrinsic'][3]),
                "k1": float(camera_data[cam]['intrinsic'][4]),
                "k2": float(camera_data[cam]['intrinsic'][5]),
                "p1": float(camera_data[cam]['intrinsic'][6]),
                "p2": float(camera_data[cam]['intrinsic'][7]),
                "offset": offset.tolist(),
                "scale": scale_factor,
                "max_bound": max_dist, 
                "rolling_shutter":camera_data[cam]['rolling_shutter'].tolist(),
                "frames":[]}

        out_test = copy.deepcopy(out_train)
        all_img = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'{str(idx).zfill(6)}.jpeg') for idx in range(start_frame, end_frame)]
        all_img_train = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'{str(idx).zfill(6)}.jpeg') for idx in range(start_frame, end_frame, step_frame)]
        
        for i, name in enumerate(all_img_train):
            path = os.sep.join(name.split(os.sep)[-3:])
            frame={"file_path":os.path.join('..','..',path), 
            "transform_matrix_start": camera_data[cam]['poses_start'][step_frame * i].tolist(), 
            "transform_matrix_end": camera_data[cam]['poses_end'][step_frame * i].tolist(), 
            }

            out_train['frames'].append(frame)

        for i, name in enumerate(all_img):
            path = os.sep.join(name.split(os.sep)[-3:])
            frame={"file_path":os.path.join('..','..',path), 
            "transform_matrix_start": camera_data[cam]['poses_start'][i].tolist(), 
            "transform_matrix_end": camera_data[cam]['poses_end'][i].tolist(), 
            }
            out_test['frames'].append(frame)

        if cam_idx == 0 and use_lidar:
            out_train['lidar'] = []
            all_lidar = [os.path.join(root_dir, 'lidar', f'{str(idx).zfill(6)}.dat') for idx in range(start_frame, end_frame + 10)] # add some lidar frames at the end as cameras see further

            for i, name in enumerate(all_lidar):
                path =  os.sep.join(name.split(os.sep)[-2:])
                frame={"file_path": os.path.join('..','..',path)}
                out_train['lidar'].append(frame)


        print('writing train camera.json...')
        with open(os.path.join(output_dir, f'cam_{cam}_train.json'), 'w') as outfile:    
            json.dump(out_train, outfile, indent=2)

        print('writing test camera.json...')
        keys_to_delete = []
        for key in out_test.keys():
            if 'ftheta' in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del out_test[key]

        with open(os.path.join(output_dir, f'cam_{cam}_test.json'), 'w') as outfile:    
            json.dump(out_test, outfile, indent=2)


@cli.command()
@click.pass_context
def nvidia(ctx, *_, **kwargs): 
    camera_data = {}
    root_dir = ctx.obj['root_dir']
    exp_name = ctx.obj['experiment_name']
    start_frame = ctx.obj['start_frame']
    end_frame = ctx.obj['end_frame']
    step_frame = ctx.obj['step_frame']
    cameras = ctx.obj['cameras']
    index_digits = ctx.obj['index_digits']
    max_dist = ctx.obj['max_dist']
    use_lidar = ctx.obj['use_lidar']

    # Check that the input arguments are valid
    assert start_frame < end_frame, "End frame index is smaller that start frame one."
    assert not os.path.exists(os.path.join(root_dir, f'ngp_configs/{exp_name}')), "Experiment with the same name already exists."
    for cam in cameras:
        assert str(cam).zfill(2) in NV_CAMERAS, "Invalid camera selected for Nvidia dataset."

    # Create the output path 
    output_dir = os.path.join(root_dir, f'ngp_configs/{exp_name}') 
    os.makedirs(output_dir)
        
    for cam in cameras: 
        camera_data[cam] = {}
        
        with open(os.path.join(root_dir, f'images/image_{str(cam).zfill(2)}', f'{str(start_frame).zfill(6)}.pkl'),'rb') as file:
            camera_metada = pickle.load(file)

        # Get the focal length, image width and height
        focal_length = camera_metada['intrinsic'][10]
        w, h = camera_metada['img_width'],  camera_metada['img_height']

        # Compute the angular field of view
        angle_x= math.atan(w/(focal_length*2))*2
        camera_data[cam]['angle'] = angle_x
        
        # Z is the up vector in the Nvidia coordinate system
        camera_data[cam]['up'] = np.array([0,0,1])

        # Load all the images and the corresponding metadata (to average the pose we neglect the selected step size as all images will be used for testing)
        all_img = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'{str(idx).zfill(index_digits)}.jpeg') for idx in range(start_frame, end_frame)]
        all_img_mask = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'mask_{str(idx).zfill(index_digits)}.png') for idx in range(start_frame, end_frame)]
        all_metadata = [img.replace('.jpeg','.pkl') for img in all_img ]

        # Resave all image masks 
        for img_mask_path in all_img_mask:
            img = Image.open(img_mask_path)
            img.save(img_mask_path.replace('mask_', 'dynamic_mask_'), bits=1,optimize=True)

        # Iterate over the poses 
        T_cam_rig = []
        poses_start = []
        poses_end = []

        # Iterate over all the frames and save their metadata
        for cam_pose in all_metadata:
            with open(cam_pose,'rb') as file:
                camera_metadata = pickle.load(file)

            T_cam_rig.append(camera_metadata['T_cam_rig'])
            T_cam_world = camera_metadata['ego_pose_s'] @ np.array(T_cam_rig)[0]
            # Iterate over the poses 
            R = T_cam_world[:3,:3] @ R_NVIDIA_NGP
            T_cam_world[:3,:3] = R
            poses_start.append(T_cam_world)

            T_cam_world = camera_metadata['ego_pose_e'] @ np.array(T_cam_rig)[0]
             # Iterate over the poses 
            R = T_cam_world[:3,:3] @ R_NVIDIA_NGP
            T_cam_world[:3,:3] = R
            poses_end.append(T_cam_world)
                   
        poses_start = np.stack(poses_start)
        poses_end = np.stack(poses_end)

        # Concatenate the bottom row
        camera_data[cam]['poses_start'] = poses_start
        camera_data[cam]['poses_end'] = poses_end
        camera_data[cam]['intrinsic'] =  camera_metadata['intrinsic']

        # Extract the rolling shutter parameters such y = a + b*x + c*y 
        a = b = 0
        c = 1.0
        camera_data[cam]['rolling_shutter'] = np.array([a, b, c])

    # Combine all the poses and compute the scaling factor and centroid, use the start timestamp pose as approximation
    all_poses = []
    for cam in cameras: 
        all_poses.append(camera_data[cam]['poses_start'])
    all_poses = np.concatenate(all_poses,axis=0)

    pose_avg, extent = average_camera_pose(all_poses)
    scale_factor = 1/ ((extent/2 + max_dist) / 8.0) # so that the max far is scaled to 5
    offset = -(pose_avg * scale_factor) + np.array([0.5,0.5,0.5]) # Instant NGP assumes that the scenes are centered at 0.5^3
    
    # Rescale and move the poses
    for cam_idx, cam in enumerate(cameras): 

        out_train={"aabb_scale":16, 
                   "n_extra_learnable_dims" : 8, 
                   "up":camera_data[cam]['up'].tolist(), 
                   "offset": offset.tolist(),
                   "scale": scale_factor,
                   "max_bound": max_dist, 
                   "w": float(camera_data[cam]['intrinsic'][2]), 
                   "h": float(camera_data[cam]['intrinsic'][3]), 
                   "cx": float(camera_data[cam]['intrinsic'][0]), 
                   "cy": float(camera_data[cam]['intrinsic'][1]), 
                   "ftheta_p0": float(camera_data[cam]['intrinsic'][4]),
                   "ftheta_p1": float(camera_data[cam]['intrinsic'][5]),
                   "ftheta_p2": float(camera_data[cam]['intrinsic'][6]),
                   "ftheta_p3": float(camera_data[cam]['intrinsic'][7]),
                   "ftheta_p4": float(camera_data[cam]['intrinsic'][8]),
                   "rolling_shutter":camera_data[cam]['rolling_shutter'].tolist(),
                   "frames":[]}

        out_test = copy.deepcopy(out_train)
        all_img = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'{str(idx).zfill(6)}.jpeg') for idx in range(start_frame, end_frame)]
        all_img_train = [os.path.join(root_dir, 'images', f'image_{str(cam).zfill(2)}', f'{str(idx).zfill(6)}.jpeg') for idx in range(start_frame, end_frame, step_frame)]
        
        for i, name in enumerate(all_img_train):
            path = os.sep.join(name.split(os.sep)[-3:])
            frame={"file_path":os.path.join('..','..',path), 
            "transform_matrix_start": camera_data[cam]['poses_start'][step_frame * i].tolist(), 
            "transform_matrix_end": camera_data[cam]['poses_end'][step_frame * i].tolist()}

            out_train['frames'].append(frame)

        for i, name in enumerate(all_img):
            path = os.sep.join(name.split(os.sep)[-3:])
            # path = name
            frame={"file_path":os.path.join('..','..',path), 
            "transform_matrix_start": camera_data[cam]['poses_start'][i].tolist(), 
            "transform_matrix_end": camera_data[cam]['poses_end'][i].tolist()}
            out_test['frames'].append(frame)

        if cam_idx == 0 and use_lidar:
            out_train['lidar'] = []

            # Find the correspondign lidar frames based on their timestampes
            camera_timestamps = pickle.load(open((os.path.join(root_dir, 'images/timestamps.pkl')),'rb'))['00']
            lidar_timestamps = np.load(os.path.join(root_dir, 'lidar/timestamps.npz'))['timestamps']
            start_timestamp = camera_timestamps[start_frame]
            end_timestamp = camera_timestamps[end_frame]
            lidar_start_idx  = np.where(lidar_timestamps > start_timestamp)[0][0] + 1
            lidar_end_idx   = np.where(lidar_timestamps < end_timestamp)[0][-1]  + 1
            all_lidar = [os.path.join(root_dir, 'lidar', f'{str(idx).zfill(6)}.dat') for idx in range(lidar_start_idx, lidar_end_idx + 50)] # add lidar at the end as cameras see further away

            for i, name in enumerate(all_lidar):
                path =  os.sep.join(name.split(os.sep)[-2:])
                # path = name
                frame={"file_path": os.path.join('..','..',path)}
                out_train['lidar'].append(frame)

        print('writing train camera.json...')
        with open(os.path.join(output_dir, f'cam_{cam}_train.json'), 'w') as outfile:    
            json.dump(out_train, outfile, indent=2)

        print('writing test camera.json...')
        keys_to_delete = []
        for key in out_test.keys():
            if 'ftheta' in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del out_test[key]

        with open(os.path.join(output_dir, f'cam_{cam}_test.json'), 'w') as outfile:    
            json.dump(out_test, outfile, indent=2)

if __name__ == '__main__':
    cli(show_default=True)