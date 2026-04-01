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

import json
import logging

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union, cast

import click
import numpy as np
import PIL.Image as PILImage
import pycolmap
import tqdm

from upath import UPath

from ncore.data.v4 import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.data_converter import FileBasedDataConverter, FileBasedDataConverterConfig
from ncore.impl.common.transformations import HalfClosedInterval, se3_inverse
from ncore.impl.data.types import (
    JsonLike,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.sensors.camera import OpenCVFisheyeCameraModel
from tools.data_converter.cli import cli


@dataclass(kw_only=True, slots=True)
class ColmapConverter4Config(FileBasedDataConverterConfig):
    """Configuration for COLMAP to NCore V4 conversion."""

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    start_time_sec: float = 0.0
    camera_prefix: str = "camera"
    include_downsampled_images: bool = False
    include_3d_points: bool = True
    colmap_dir: str = "sparse/0"
    images_dir: str = "images"
    masks_dir: Optional[str] = None
    generic_meta_data: dict[str, JsonLike] = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class ColmapCamera:
    """Intermediate class that stores the colmap camera information for converting to NCore."""

    camera_id: str
    colmap_camera: pycolmap.Camera
    image_path: UPath
    downsample_factor: int = 1
    image_names: list[str] = field(default_factory=list)
    T_ref_camera_list: list[np.ndarray] = field(default_factory=list)
    reference_frame: str = "world"

    @property
    def n_images(self) -> int:
        return len(self.image_names)

    @property
    def timestamps_us(self, start_time_sec: float = 0.0) -> np.ndarray:
        return (1e6 * (start_time_sec + np.linspace(0.0, self.n_images - 1, self.n_images))).astype(np.uint64)

    @property
    def T_camera_refs(self) -> np.ndarray:
        T_ref_cameras = np.stack(self.T_ref_camera_list, axis=0)

        # Convert extrinsics to camera-to-reference frame.
        # NOTE: colmap already assumes OpenCV convention
        T_camera_refs = se3_inverse(T_ref_cameras)
        return T_camera_refs[:, :4, :4].astype(np.float32)

    @property
    def camera_model(self) -> Union[OpenCVPinholeCameraModelParameters, OpenCVFisheyeCameraModelParameters]:
        camera = self.colmap_camera

        assert camera.camera_type in [0, 1, 2, 3, 4, 5], f"Unsupported camera type: {camera.camera_type}"

        width = round(camera.width / self.downsample_factor)
        height = round(camera.height / self.downsample_factor)

        # Resolution check for downsampled images
        if self.downsample_factor != 1:
            with PILImage.open(str(self.image_path / self.image_names[0])) as img:
                if img.width != width or img.height != height:
                    logging.getLogger(__name__).warning(
                        f"Unexpected resolution: {img.width} {img.height}. Expected {width} {height}"
                    )
                    height = img.height
                    width = img.width

        focal_length = np.array(
            [camera.fx / self.downsample_factor, camera.fy / self.downsample_factor], dtype=np.float32
        )
        principal_point = np.array(
            [camera.cx / self.downsample_factor, camera.cy / self.downsample_factor], dtype=np.float32
        )

        # 5: OpenCVFisheye -> OpenCVFisheyeCameraModelParameters
        if camera.camera_type == 5:
            resolution = np.array([width, height], dtype=np.uint64)
            radial_coeffs = np.array([camera.k1, camera.k2, camera.k3, camera.k4], dtype=np.float32)
            max_angle = OpenCVFisheyeCameraModel.compute_max_angle(
                resolution, focal_length, principal_point, radial_coeffs
            )
            return OpenCVFisheyeCameraModelParameters(
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
                external_distortion_parameters=None,
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                max_angle=max_angle,
            )

        # 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL, 4: OpenCV
        radial_coeffs = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        tangential_coeffs = np.array([0, 0], dtype=np.float32)

        if camera.camera_type > 1:
            radial_coeffs[0] = camera.k1
        if camera.camera_type > 2:
            radial_coeffs[1] = camera.k2
        if camera.camera_type == 4:
            tangential_coeffs[0] = camera.p1
            tangential_coeffs[1] = camera.p2

        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([width, height], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            external_distortion_parameters=None,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
        )


class ColmapDataConverter(FileBasedDataConverter):
    """
    NVIDIA-specific data conversion from COLMAP reconstructions to NCore V4 using the V4 components/writer APIs.
    """

    def __init__(self, config: ColmapConverter4Config):
        """Initializes the converter with the given configuration."""

        super().__init__(config)

        self.camera_prefix = config.camera_prefix
        self.start_time_sec = config.start_time_sec
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )
        self.store_sequence_meta: bool = config.store_sequence_meta

        # Downsampled images in folders images_2, images_4, etc will be included as additional cameras
        self.include_downsampled_images = config.include_downsampled_images
        self.include_3d_points = config.include_3d_points
        self.colmap_dir = config.colmap_dir
        self.images_dir = config.images_dir
        self.masks_dir = config.masks_dir
        self.generic_meta_data: dict[str, JsonLike] = config.generic_meta_data
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: ColmapConverter4Config) -> list[str]:
        """Get sequence paths from config.

        If root_dir is a directory, return all subdirectories as sequences. If root_dir is a single sequence, return it as the only sequence."""

        if str(config.root_dir).endswith("/"):
            return [str(p) for p in UPath(config.root_dir).iterdir() if p.is_dir()]
        else:
            return [str(UPath(config.root_dir))]

    @staticmethod
    def from_config(config: ColmapConverter4Config) -> ColmapDataConverter:
        """Create a ColmapDataConverter instance from a configuration object."""
        return ColmapDataConverter(config)

    def convert_sequence(self, sequence_id: str) -> None:
        """
        Runs the conversion of a single sequence
        """

        sequence_path = UPath(sequence_id)

        self.logger.info(f"Processing sequence: {sequence_path}")

        self.sequence_path = sequence_path
        self.sequence_name = sequence_path.name

        bin_path = self.sequence_path / self.colmap_dir

        image_path = self.sequence_path / self.images_dir
        self.scene_manager = pycolmap.SceneManager(str(bin_path), image_path=str(image_path))
        self.scene_manager.load()

        self.cameras = self.populate_camera_data(
            parent_dir=self.sequence_path,
            camera_prefix=self.camera_prefix,
            downsample=self.include_downsampled_images,
            images_dir=self.images_dir,
        )

        camera_ids = list(self.cameras.keys())
        lidar_id = "virtual_lidar" if len(self.scene_manager.points3D) > 0 and self.include_3d_points else None
        self.logger.info(f"Adding cameras: {camera_ids} and lidar: {lidar_id}")

        self.component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=[lidar_id] if lidar_id is not None else [],
            radar_ids=[],  # No radars for now
            profile=self.component_group_profile,
        )

        # Use this to calculate the time span
        max_poses = np.max([camera.n_images for camera in self.cameras.values()])

        # Calculate the full timespan of the sequence
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            int(1e6 * self.start_time_sec),
            int(1e6 * (self.start_time_sec + max_poses)),
        )

        # Create main store writer
        self.store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / self.sequence_name,
            store_base_name=self.sequence_name,
            sequence_id=self.sequence_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data=self.generic_meta_data,
        )

        # Create poses component
        self.poses_writer = self.store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.poses_component_group,
        )

        # Create intrinsics component
        self.intrinsics_writer = self.store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.intrinsics_component_group,
        )

        # Create masks component
        self.masks_writer = self.store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.masks_component_group,
        )

        # Create cuboids component, explicitly store abscent observations
        self.store_writer.register_component_writer(
            CuboidsComponent.Writer,
            "default",
            self.component_groups.cuboid_track_observations_component_group,
        ).store_observations([])

        ## Decode 3D Points as a single lidar frame
        if lidar_id is not None:
            self.decode_lidars(lidar_id)

        ## Decode and store camera frames
        self.decode_cameras()

        # Store per-shard meta data / final success state / close file
        ncore_4_paths = self.store_writer.finalize()

        # Output sequence meta file if requested
        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = self.output_dir / self.sequence_name / f"{sequence_component_reader.sequence_id}.json"

            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    def decode_lidars(self, lidar_id) -> None:
        """Decodes COLMAP 3D points as a single lidar frame."""

        self.poses_writer.store_static_pose(
            source_frame_id=lidar_id,
            target_frame_id="world",
            pose=np.eye(4, dtype=np.float32),
        )

        lidar_writer = self.store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            component_instance_name=lidar_id,
            group_name=self.component_groups.lidar_component_groups.get(lidar_id),
        )

        xyz = self.scene_manager.points3D.astype(np.float32)
        distance = np.linalg.norm(xyz, axis=1)  # N
        # Filter out points with zero (or near-zero) distance to avoid division by zero
        valid_mask = distance > 1e-6
        distance = distance[valid_mask]
        direction = xyz[valid_mask] / distance[valid_mask, np.newaxis]
        num_points = direction.shape[0]
        start_timestamp_us = np.uint64(self.start_time_sec * 1e6)

        # Serialize lidar frame
        lidar_writer.store_frame(
            # Non-motion-compensated per-ray 3D directions (float32, [N, 3])
            direction=direction,
            # Per-point timestamp in microseconds (uint64, [N])
            timestamp_us=np.full((num_points,), start_timestamp_us, dtype=np.uint64),
            # Per-point model element indices (uint16, [N, 2])
            model_element=None,
            # Per-point distance (two returns, [2, N])
            distance_m=distance[np.newaxis, :],
            # Per-point intensity normalized to [0.0, 1.0] (float32, [2, N])
            intensity=np.zeros((1, num_points), dtype=np.float32),
            # Frame start/end timestamps (uint64, [2])
            frame_timestamps_us=np.array([start_timestamp_us, start_timestamp_us], dtype=np.uint64),
            # Store COLMAP SfM point colors as generic data (uint8, [N, 3])
            generic_data={
                "rgb": self.scene_manager.point3D_colors,  # uint8
            },
            generic_meta_data={},
        )

    def populate_camera_data(
        self, parent_dir: UPath, camera_prefix: str, downsample: bool, images_dir: str = "images"
    ) -> dict[str, ColmapCamera]:
        """Populates the camera data from the COLMAP scene, including extrinsics and intrinsics

        Returns a dictionary of ColmapCamera instances indexed by the NCore camera IDs."""

        cameras: dict[str, ColmapCamera] = {}

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.scene_manager.images

        for k in imdata:
            img: pycolmap.Image = imdata[k]
            rot = img.R()
            trans = img.tvec.reshape(3, 1)
            T_ref_camera = np.concatenate(
                [np.concatenate([rot, trans], 1), np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
            )
            ncore_camera_id = camera_prefix + str(imdata[k].camera_id)

            camera = self.scene_manager.cameras[imdata[k].camera_id]

            if ncore_camera_id not in cameras:
                cameras[ncore_camera_id] = ColmapCamera(
                    camera_id=ncore_camera_id,
                    colmap_camera=self.scene_manager.cameras[imdata[k].camera_id],
                    image_path=parent_dir / images_dir,
                )
            cameras[ncore_camera_id].T_ref_camera_list.append(T_ref_camera)
            cameras[ncore_camera_id].image_names.append(imdata[k].name)

        if downsample:
            # Add extra cameras if we are downsampling and data exists.
            for camera_id, camera in self.scene_manager.cameras.items():
                assert isinstance(camera, pycolmap.Camera)
                for downsample_factor in [2, 4, 8]:
                    reference_camera_id = camera_prefix + str(camera_id)
                    ncore_camera_id = camera_prefix + str(camera_id) + "_" + str(downsample_factor)
                    image_root = "images_" + str(downsample_factor)
                    if not (parent_dir / image_root).exists():
                        self.logger.warning(f"Skipping missing downsampled image directory: {image_root}")
                        continue
                    image_names = cameras[reference_camera_id].image_names
                    cameras[ncore_camera_id] = ColmapCamera(
                        camera_id=ncore_camera_id,
                        colmap_camera=self.scene_manager.cameras[imdata[k].camera_id],
                        image_path=parent_dir / image_root,
                        reference_frame=reference_camera_id,
                        T_ref_camera_list=[np.eye(4)] * len(image_names),
                        image_names=image_names,
                        downsample_factor=downsample_factor,
                    )

        return cameras

    def _find_mask_path(self, image_path: UPath, image_name: str) -> Optional[UPath]:
        """Find a per-image mask file using multiple conventions.

        Checks the following locations in priority order:

        0. ``<sequence_dir>/<masks_dir>/<stem>.png`` or ``<sequence_dir>/<masks_dir>/<image_filename>``
           — explicit masks directory (from config)
        1. ``<image_dir>/<stem>_mask.png`` — co-located mask
        2. ``<sequence_dir>/masks/<image_filename>`` — separate masks directory

        Parameters
        ----------
        image_path : UPath
            Full path to the image file.
        image_name : str
            Filename of the image (e.g. ``img_00.png``).

        Returns
        -------
        Optional[UPath]
            Path to the mask file if found, otherwise ``None``.
        """
        stem = image_path.stem

        # Convention 0: explicit masks directory (from config)
        if self.masks_dir is not None:
            mask_path = self.sequence_path / self.masks_dir / f"{stem}.png"
            if mask_path.exists():
                return mask_path
            mask_path = self.sequence_path / self.masks_dir / image_name
            if mask_path.exists():
                return mask_path

        # Convention 1: <image_dir>/<stem>_mask.png
        mask_path = image_path.parent / f"{stem}_mask.png"
        if mask_path.exists():
            return mask_path

        # Convention 2: <sequence_dir>/masks/<image_filename>
        mask_path = self.sequence_path / "masks" / image_name
        if mask_path.exists():
            return mask_path

        return None

    def decode_cameras(self) -> None:
        """Decodes camera frames, intrinsics, and masks from the COLMAP scene."""

        for camera_ncore_id, colmap_camera in self.cameras.items():
            self.poses_writer.store_dynamic_pose(
                source_frame_id=camera_ncore_id,
                target_frame_id=colmap_camera.reference_frame,
                poses=colmap_camera.T_camera_refs,
                timestamps_us=colmap_camera.timestamps_us,
                require_sequence_time_coverage=False,  # Some cameras may have more poses than others
            )

            # Create camera sensor writer
            camera_writer = self.store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_ncore_id,
                group_name=self.component_groups.camera_component_groups.get(camera_ncore_id),
            )

            # Store intrinsics
            self.intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_ncore_id,
                camera_model_parameters=colmap_camera.camera_model,
            )

            # Store empty static masks (per-image masks are stored as per-frame generic data below)
            self.masks_writer.store_camera_masks(
                camera_id=camera_ncore_id,
                mask_images={},
            )
            timestamps_us = colmap_camera.timestamps_us

            masks_found = 0
            for continuous_frame_index in tqdm.tqdm(range(colmap_camera.n_images), desc=f"Decoding {camera_ncore_id}"):
                image_name = colmap_camera.image_names[continuous_frame_index]
                image_path = colmap_camera.image_path / image_name

                if (file_extension := image_name.split(".")[-1].lower()) == "jpg":
                    # PIL in ncore uses jpeg instead of jpg
                    file_extension = "jpeg"

                ## Load current camera's image
                with image_path.open("rb") as f:
                    image_bytes = f.read()

                # single pose and timestamp for global shutter
                frame_timestamps_us = np.array(
                    [timestamps_us[continuous_frame_index], timestamps_us[continuous_frame_index]]
                ).astype(np.uint64)

                generic_data: dict[str, np.ndarray] = {}
                generic_meta_data: dict[str, JsonLike] = {}

                # Check for a per-image mask file (grayscale, stored as generic frame data)
                mask_path = self._find_mask_path(image_path, image_name)
                if mask_path is not None:
                    mask_array = np.array(PILImage.open(str(mask_path)).convert("L"), dtype=np.uint8)
                    generic_data["mask"] = mask_array
                    masks_found += 1

                camera_writer.store_frame(
                    image_binary_data=image_bytes,
                    image_format=file_extension,
                    frame_timestamps_us=frame_timestamps_us,
                    generic_data=generic_data,
                    generic_meta_data=generic_meta_data,
                )

            if masks_found > 0:
                self.logger.info(f"Found {masks_found}/{colmap_camera.n_images} per-image masks for {camera_ncore_id}")


@cli.command()
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type",
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help="""Output profile, one of:
        - "default": All components defaults or overrides
        - "separate-sensors": Each sensor gets its own group named "<sensor_id>", remaining components use overrides
        - "separate-all": Each component type gets its own group named after the component type, e.g. "poses", "intrinsics", respecting overwrites if provided""",
)
@click.option(
    "store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True, help="Generate sequence meta-data?"
)
@click.option(
    "--start-time-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    default=0.0,
    help="Start time (in seconds).",
)
@click.option(
    "--camera-prefix",
    type=str,
    default="camera",
    help="Camera name prefix (for integer camera_ids). Default is 'camera'.",
)
@click.option(
    "--include-downsampled-images/--no-include-downsampled-images",
    is_flag=True,
    default=True,
    help="Include downsampled images as additional cameras. Images should be in folders such as `images_2`, etc.",
)
@click.option(
    "--include-3d-points/--no-include-3d-points",
    is_flag=True,
    default=True,
    help="Include 3D Points as a lidar sensor",
)
@click.option(
    "--colmap-dir",
    type=str,
    default="sparse/0",
    show_default=True,
    help="Path to the COLMAP sparse reconstruction directory (relative to sequence root).",
)
@click.option(
    "--images-dir",
    type=str,
    default="images",
    show_default=True,
    help="Path to the images directory (relative to sequence root).",
)
@click.option(
    "--masks-dir",
    type=str,
    default=None,
    help="Path to the masks directory (relative to sequence root). If not set, masks are discovered via conventions.",
)
@click.pass_context
def colmap_v4(ctx, *_, **kwargs):
    """Colmap-specific data conversion (V4 format)"""

    # Extend base config with command-specific options
    config = ColmapConverter4Config(**{**vars(ctx.obj), **kwargs})

    ColmapDataConverter.convert(config)


# ---------------------------------------------------------------------------
# ScanNet++ subcommand
# ---------------------------------------------------------------------------


def _discover_scannetpp_scenes(root_dir: UPath) -> list[UPath]:
    """Discover ScanNet++ scenes under *root_dir*.

    A valid scene is a subdirectory containing ``dslr/colmap/``.
    """
    scenes = []
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and (child / "dslr" / "colmap").is_dir():
            scenes.append(child)
    return scenes


def _build_scannetpp_split_metadata(
    scene_path: UPath,
    image_names: list[str],
    camera_id: str,
    start_time_sec: float,
) -> dict[str, JsonLike]:
    """Build train/test split metadata from ScanNet++ ``train_test_lists.json``.

    Returns a dict suitable for sequence-level ``generic_meta_data``.
    """
    train_test_path = scene_path / "dslr" / "train_test_lists.json"
    if not train_test_path.exists():
        logging.getLogger(__name__).warning(f"No train_test_lists.json found at {train_test_path}")
        return {}

    with train_test_path.open("r") as f:
        train_test = json.load(f)

    train_set = set(train_test.get("train", []))
    test_set = set(train_test.get("test", []))

    # Associate image file names with virtual frame timestamps
    sorted_names = sorted(image_names)
    name_to_ts = {name: int(1e6 * (start_time_sec + i)) for i, name in enumerate(sorted_names)}

    train_filenames = [n for n in sorted_names if n in train_set]
    test_filenames = [n for n in sorted_names if n in test_set]

    camera_splits: dict[str, JsonLike] = {}
    if train_filenames:
        camera_splits["train"] = {
            "frame_timestamps_us": [name_to_ts[n] for n in train_filenames],
            "source_filenames": cast(List[JsonLike], train_filenames),
        }
    if test_filenames:
        camera_splits["test"] = {
            "frame_timestamps_us": [name_to_ts[n] for n in test_filenames],
            "source_filenames": cast(List[JsonLike], test_filenames),
        }

    return {
        "source_dataset": "scannetpp",
        "scene_id": scene_path.name,
        "splits": {camera_id: camera_splits} if camera_splits else {},
    }


@cli.command()
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type.",
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help="Output component group profile.",
)
@click.option(
    "--include-3d-points/--no-include-3d-points",
    is_flag=True,
    default=True,
    help="Include COLMAP SfM point cloud as virtual lidar.",
)
@click.pass_context
def scannetpp_v4(ctx, **kwargs):
    """ScanNet++ dataset conversion (V4 format).

    Converts ScanNet++ DSLR scenes to NCore V4 by configuring the COLMAP
    converter for the ScanNet++ directory layout.  Uses the resized fisheye
    images (``dslr/resized_images/``) with the COLMAP OPENCV_FISHEYE camera
    model.

    Expects ``--root-dir`` to point either to a single scene directory or
    to a parent directory containing multiple scene subdirectories.
    """

    _logger = logging.getLogger(__name__)
    base = ctx.obj
    root_dir = UPath(base.root_dir)

    # Discover scenes
    scenes = _discover_scannetpp_scenes(root_dir)
    if not scenes:
        if (root_dir / "dslr" / "colmap").is_dir():
            scenes = [root_dir]
        else:
            raise click.ClickException(f"No ScanNet++ scenes found in {root_dir}")

    _logger.info(f"Found {len(scenes)} ScanNet++ scene(s)")

    for scene_path in scenes:
        scene_id = scene_path.name

        # Check has_masks from train_test_lists.json
        train_test_path = scene_path / "dslr" / "train_test_lists.json"
        has_masks = False
        if train_test_path.exists():
            with train_test_path.open("r") as f:
                has_masks = json.load(f).get("has_masks", False)

        masks_dir: Optional[str] = "dslr/resized_anon_masks" if has_masks else None

        # ScanNet++ DSLR has exactly 1 COLMAP camera -> NCore ID "dslr1"
        camera_prefix = "dslr"
        images_dir = "dslr/resized_images"
        primary_image_dir = scene_path / images_dir
        primary_image_names = sorted(f.name for f in primary_image_dir.iterdir() if f.is_file())

        generic_meta_data = _build_scannetpp_split_metadata(
            scene_path=scene_path,
            image_names=primary_image_names,
            camera_id=f"{camera_prefix}1",
            start_time_sec=kwargs.get("start_time_sec", 0.0),
        )

        config = ColmapConverter4Config(
            root_dir=str(scene_path),
            output_dir=base.output_dir,
            verbose=base.verbose,
            debug=base.debug,
            debug_port=base.debug_port,
            no_cameras=base.no_cameras,
            camera_ids=base.camera_ids,
            no_lidars=base.no_lidars,
            lidar_ids=base.lidar_ids,
            no_radars=base.no_radars,
            radar_ids=base.radar_ids,
            colmap_dir="dslr/colmap",
            images_dir=images_dir,
            masks_dir=masks_dir,
            camera_prefix=camera_prefix,
            include_downsampled_images=False,
            generic_meta_data=generic_meta_data,
            **kwargs,
        )

        _logger.info(f"Converting scene: {scene_id}")
        ColmapDataConverter.convert(config)


if __name__ == "__main__":
    cli(show_default=True)
