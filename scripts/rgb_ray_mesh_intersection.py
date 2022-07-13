# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import argparse
import point_cloud_utils as pcu
import numpy as np
import struct
import random
import os
import pickle
import glob
from PIL import Image

import sys
sys.path.append('./')
from lib import image_to_world_ray
from src.common import NV_CAMERAS, WAYMO_CAMERAS

def generate_colored_pclouds(args, cam_id):
    vertices, faces = pcu.load_mesh_vf(os.path.join(args.root_dir,'reconstructed_surface', "reconstructed_mesh.ply" ))
    output_dir = os.path.join(args.root_dir, 'color_point_clouds', f'image_{cam_id}') if not args.output_dir else os.path.join(args.output_dir, f'image_{cam_id}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for data_path in \
            sorted(glob.glob(os.path.join(args.root_dir, f'images/image_{cam_id}', f"{'?'*args.index_digits}.pkl")))[args.start_at:args.stop_at:args.step]:
        # Load the image metadata and compute the world rays (considering rolling shutter parameters)
        with open(data_path, 'rb') as img_file:
            metadata = pickle.load(img_file)

        img_width = metadata['img_width']
        img_height = metadata['img_height']

        # Remove the bottom part of the image as it contains the ego car
        u = np.tile(np.arange(int(img_width)), int(img_height))
        v = np.repeat(np.arange(int(img_height)), int(img_width))
        image_points = np.concatenate([u.reshape(-1,1), v.reshape(-1,1)], axis=1).astype(np.float64)
        img_world_rays = image_to_world_ray(image_points, metadata) # Nx6 with columns denoting x_s, y_s, z_s, x_e, y_e, z_e
        ray_o = np.ascontiguousarray(img_world_rays[:,:3])
        ray_d = np.ascontiguousarray(img_world_rays[:,3:6])

        # Save the rays in dat file if save_rays selected
        if args.save_rays:
            img_rays_flat = img_world_rays.flatten()
            with open(os.path.join(output_dir, data_path.split(os.sep)[-1].replace('.pkl', '.dat')),'wb') as f:
                f.write(struct.pack('<i', img_world_rays.shape[0]))
                f.write(struct.pack('<i', img_world_rays.shape[1]))
                f.write(struct.pack('<%sf' % img_rays_flat.size, *img_rays_flat))

        rgb_values = np.array(Image.open(data_path.replace('.pkl','.jpeg')).convert('RGB')).reshape([-1, 3])
        fid, bc, t = pcu.ray_mesh_intersection(vertices, faces, ray_o, ray_d)
        valid_rays = np.isfinite(t)

        pcu.save_mesh_vc(os.path.join(output_dir, data_path.split(os.sep)[-1].replace('.pkl','.ply')), 
                pcu.interpolate_barycentric_coords(faces, fid[valid_rays], bc[valid_rays], vertices).astype(np.float32), rgb_values[valid_rays].astype(np.float32) / 255.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="Path to the raw data", required=True)
    parser.add_argument("--dataset", type=str, help="Name of the dataset", choices=['nvidia-deepmap', 'waymo', 'nvidia-maglev'], required=True)
    parser.add_argument("--cam_id", type=str, help="Camera ID to be used for projection. If not specified random camera will be used", default= '-1')
    parser.add_argument('--output_dir', type=str, default = '', help="Output path. If not specified the default path 'root_dir/colored_point_clouds/' will be used")
    parser.add_argument("--start_at", type=int, default=0, help="Idx of the first RGB frame to be used (default 0 so all images will be used)")
    parser.add_argument("--stop_at", type=int, default=-1, help="Idx of the final RGB frame to be used (default -1 so all images will be used)")
    parser.add_argument("--step", type=int, default=1, help="Frame step (Default 1 so each frame will be used)")
    parser.add_argument("--save_rays", action='store_true', help="If selected 3D RGB rays will be saved to dat files.")
    parser.add_argument("--index_digits", type=int, help="The number of integer digits to pad counters in output filenames to", default=6)
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.root_dir,'reconstructed_surface', "reconstructed_mesh.ply")), "Mesh of the reconstructed surface does not exist!"
    
    # Select the correct maps
    CAM_IDS = NV_CAMERAS if args.dataset.startswith('nvidia') else WAYMO_CAMERAS
    if args.cam_id != '-1':
        if args.cam_id not in CAM_IDS:
            print(f'{args.cam_id} is not a valid cam id for dataset {args.dataset}. A random cam will be selected instead.')
            cam_id =  random.choice(CAM_IDS)
        else:
            cam_id = args.cam_id
    else:
        print('Camera id not provided. A random camera will be selected.')
        cam_id =  random.choice(CAM_IDS)

    assert os.path.isdir(os.path.join(args.root_dir,'images', f"image_{cam_id}" )), f"Image folder for the selected camera {cam_id} does not exist!"

    generate_colored_pclouds(args, cam_id)