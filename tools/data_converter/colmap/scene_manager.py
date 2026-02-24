# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import pycolmap

from PIL import Image
from upath import UPath

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters, ShutterType


class ColmapSceneManager(pycolmap.SceneManager):
    """COLMAP camera intrinsics and extrinsics loader.

    Minor NCore-specific extension to the third_party Python COLMAP loader:
    https://github.com/trueprice/pycolmap
    """

    def process(self, parent_dir: UPath, camera_prefix: str, start_time_sec: float, downsample: bool):
        """
        Applies NCore-specific postprocessing to the loaded pose data.
        """
        self.load_cameras()
        self.load_images()
        self.load_points3D()

        self.camera_info: dict[str, dict] = {}

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im: pycolmap.Image = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        # NOTE: colmap already assumes OpenCV convention
        c2w_mats = np.linalg.inv(w2c_mats)
        poses = c2w_mats[:, :4, :4].astype(np.float64)

        # Image names and camera ids from COLMAP
        image_names = [imdata[k].name for k in imdata]
        camera_ids = np.array([imdata[k].camera_id for k in imdata], dtype=np.int32)

        # timestamps are arbitrary.  We use 1 FPS to keep things simple.
        timestamps_us = np.array([1_000_000 * (start_time_sec + i) for i in range(len(image_names))]).astype(np.uint64)

        for camera_id, camera in self.cameras.items():
            assert isinstance(camera, pycolmap.Camera)

            # Get distortion parameters.
            assert camera.camera_type in [0, 1, 2, 3, 4]
            radial_coeffs = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            tangential_coeffs = np.array([0, 0], dtype=np.float32)

            # 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL, 4: OpenCV, 5: OpenCVFisheye
            if camera.camera_type > 1:
                radial_coeffs[0] = camera.k1
            if camera.camera_type > 2:
                radial_coeffs[1] = camera.k2
            if camera.camera_type == 4:
                tangential_coeffs[0] = camera.p1
                tangential_coeffs[1] = camera.p2
            if camera.camera_type == 5:
                # TODO:
                raise NotImplementedError

            # the names of images associated with this camera id
            camera_image_names = [name for i, name in enumerate(image_names) if camera_ids[i] == camera_id]

            # Treat downsampled images as separate cameras with adjusted intrinsics
            downsample_factors = [1, 2, 4, 8] if downsample else [1]
            for downsample_factor in downsample_factors:
                camera_name = camera_prefix + str(camera_id)
                if downsample_factor != 1:
                    camera_name = camera_prefix + str(camera_id) + "_" + str(downsample_factor)

                width = round(camera.width / downsample_factor)
                height = round(camera.height / downsample_factor)

                image_root = "images" if downsample_factor == 1 else "images_" + str(downsample_factor)

                if not (parent_dir / image_root).exists():
                    print(f"Skipping missing image directory: {image_root}")
                    continue

                # This is kind of a hack, but sometimes COLMAP downsampled images have a resolution different from what
                # is expected. Specifically 0.5 does not always round up.
                if downsample_factor != 1:
                    img = Image.open(parent_dir / image_root / camera_image_names[0])
                    if img.width != width or img.height != height:
                        print(f"Unexpected resolution: {img.width} {img.height}. Expected {width} {height}")
                        height = img.height
                        width = img.width

                focal_length = np.array(
                    [camera.fx / downsample_factor, camera.fy / downsample_factor], dtype=np.float32
                )
                principal_point = np.array(
                    [camera.cx / downsample_factor, camera.cy / downsample_factor], dtype=np.float32
                )

                camera_model = OpenCVPinholeCameraModelParameters(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=ShutterType.GLOBAL,
                    external_distortion_parameters=None,
                    principal_point=principal_point,
                    focal_length=focal_length,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
                )

                self.camera_info[camera_name] = {
                    "camera_model": camera_model,
                    "image_names": camera_image_names,
                    "timestamps_us": timestamps_us[camera_ids == camera_id],
                    "image_root": image_root,
                }

        return poses, timestamps_us
