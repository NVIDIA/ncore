# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os
import logging
import glob
import pickle

from typing import Tuple

from PIL import Image
import numpy as np

from src.py.common.nvidia_utils import (compute_ftheta_parameters, compute_fw_polynomial)


def downsample_ftheta(intrinsic_array, inverse_downsample_factor) -> Tuple[np.array, Tuple[int, int]]:
    ''' Decode relevant / non-'derived' camera model parameters (backward poly only, forward-poly + ray limits are derived)
        and perform camera model downsampling '''

    # V1 camera model conventions [note, very bad / fragile representation of parameters, will be more explicit in V2]
    # [[cx, cy, width, height, bwpoly], fwpoly, max_ray_distortion, dmax_ray_distortion, max_angle]
    # - 4th-degree polynominals -> total length = 17
    # - 5th-degree polynominals -> total length = 19
    if len(intrinsic_array) == 17:
        poly_degree = 4
    elif len(intrinsic_array) == 19:
        poly_degree = 5
    else:
        raise ValueError('unsupported polynomial degree')

    cxcy = np.array(intrinsic_array[0:2])
    img_width = int(intrinsic_array[2])
    img_height = int(intrinsic_array[3])
    bw_poly = np.polynomial.Polynomial(intrinsic_array[4:(5 + poly_degree)])

    ## Downsample resolution

    # check if requested downsampling is supported by model's resolution (not resulting in fractional resolutions)
    assert img_width % inverse_downsample_factor == 0 and img_height % inverse_downsample_factor == 0, 'incompatible downsample factor for image resolution'

    img_width_downsampled = img_width // inverse_downsample_factor
    img_height_downsampled = img_height // inverse_downsample_factor

    ## Downsample principal point location by transforming it in the downsampled image
    cxcy_downsampled = cxcy / inverse_downsample_factor

    ## Downsample polynomial by substituting the input pixel domain transformation (backwards polynomial is a pixel-distance to angle map, so the domain needs to be scaled)
    downsample_pixel_map = np.polynomial.Polynomial([0.0, inverse_downsample_factor])
    bw_poly_downsampled = bw_poly(downsample_pixel_map)

    # sanity check that the new transformation is a correctly downscaled version of the original polynomial
    assert bw_poly(1.0) == bw_poly_downsampled(1.0 / inverse_downsample_factor), 'computed invalid polynomial map'

    ## Update intrinsics and estimate derived parameters + prepare outputs in V1 format
    intrinsic_array_downsampled = np.hstack(
        (cxcy_downsampled, img_width_downsampled, img_height_downsampled, bw_poly_downsampled.coef))

    # Estimate the forward polynomial and other F-theta parameters (replicating V1 outputs)
    fw_poly_coeff = compute_fw_polynomial(intrinsic_array_downsampled)
    max_ray_distortion, max_angle = compute_ftheta_parameters(
        np.concatenate((intrinsic_array_downsampled, fw_poly_coeff)))
    intrinsic_array_downsampled = np.concatenate(
        (intrinsic_array_downsampled, fw_poly_coeff, max_ray_distortion, max_angle))

    return intrinsic_array_downsampled.astype(np.float32), (int(img_width_downsampled), int(img_height_downsampled))


@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed DSAI sequence to update', required=True)
@click.option('--cam-id', type=str, help='Camera to be used', default='00')
@click.option('--downsample-factor',
              'inverse_downsample_factor',
              type=click.IntRange(min=2, max_open=True),
              default=2,
              help='*Inverse* downsample factor')
@click.option('--downsample-images',
              is_flag=True,
              default=False,
              help='Downsample image data in addition to camera model')
def dsai_downsample_camera(root_dir: str, cam_id: int, inverse_downsample_factor: int, downsample_images: bool):
    ''' Downsamples FTheta Camera Model '''

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info(f'Downsampling camera {cam_id} with downsample_factor = {1.0 / inverse_downsample_factor} <{root_dir}>')

    camera_meta_paths = sorted(glob.glob(os.path.join(root_dir, f'images/image_{cam_id}/*.pkl')))

    # We need to update all the frames of the selected camera meta files, but intrinsics are the same, so only load once
    with open(camera_meta_paths[0], 'rb') as f:
        meta = pickle.load(f)

        assert meta['camera_model'] == 'f_theta', 'currently only support downsampling of ftheta cameras'

        intrinsic_array_downsampled, (img_width_downsampled,
                                      img_height_downsampled) = downsample_ftheta(meta['intrinsic'],
                                                                                  inverse_downsample_factor)

    # Update all intrinsics in camera-associated meta files
    for camera_meta_path in camera_meta_paths:
        with open(camera_meta_path, 'rb') as f:
            meta = pickle.load(f)

        meta['intrinsic'] = intrinsic_array_downsampled
        # Note: seems like V1 format expects floats, not integers [these parameters are redundant anyway]
        meta['img_width'] = float(img_width_downsampled)
        meta['img_height'] = float(img_height_downsampled)

        with open(camera_meta_path, 'wb') as f:
            pickle.dump(meta, f)

    if downsample_images:
        for image_path in set(
                glob.glob(os.path.join(root_dir, f'images/image_{cam_id}/*.jpg')) +
                glob.glob(os.path.join(root_dir, f'images/image_{cam_id}/*.jpeg')) +
                glob.glob(os.path.join(root_dir, f'images/image_{cam_id}/*.png'))):
            img = Image.open(image_path)
            img_width, img_height = img.size[0], img.size[1]
            assert img_width % inverse_downsample_factor == 0 and img_height % inverse_downsample_factor == 0, 'incompatible downsample factor for image resolution'
            img = img.resize((img_width // inverse_downsample_factor, img_height // inverse_downsample_factor),
                             Image.Resampling.LANCZOS)
            img.save(image_path)


if __name__ == "__main__":
    dsai_downsample_camera()
