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

import dataclasses
import io
import tempfile
import unittest

from typing import Dict, Literal, Tuple, cast

import numpy as np
import PIL.Image as PILImage

from parameterized import parameterized, parameterized_class
from scipy.spatial.transform import Rotation as R
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.types import (
    BBox3,
    BivariateWindshieldModelParameters,
    CuboidTrackObservation,
    JsonLike,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    PointCloud,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    ComponentReader,
    ComponentWriter,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PointCloudsComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)


@parameterized_class(
    ("store_type"),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestData4Reload(unittest.TestCase):
    """Test to verify functionality of V4 data writer + loader"""

    store_type: Literal["itar", "directory"]

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_reload(self, open_consolidated: bool):
        """Test to make sure serialized data is faithfully reloaded"""

        tempdir = tempfile.TemporaryDirectory()

        ## Create reference sequence
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(tempdir.name),
            store_base_name=(ref_sequence_id := "some-sequence-name"),
            sequence_id=ref_sequence_id,
            sequence_timestamp_interval_us=(
                ref_sequence_timestamp_interval_us := HalfClosedInterval(int(0 * 1e6), int(1 * 1e6) + 1)
            ),
            store_type=self.store_type,
            generic_meta_data=cast(Dict[str, JsonLike], ref_generic_sequence_meta_data := {"some": 1, "key": 1.2}),
        )

        # Store pose / extrinsics
        T_world_world_global = np.eye(4, dtype=np.float64)

        T_rig_worlds = np.stack(
            [
                np.block(
                    [
                        [R.from_euler("xyz", [0, 1, 2], degrees=True).as_matrix(), np.array([1, 2, 3]).reshape((3, 1))],
                        [np.array([0, 0, 0, 1])],
                    ]
                ),
                np.block(
                    [
                        [
                            R.from_euler("xyz", [0, 1.1, 2.2], degrees=True).as_matrix(),
                            np.array([1.1, 2.2, 3.3]).reshape((3, 1)),
                        ],
                        [np.array([0, 0, 0, 1])],
                    ]
                ),
                np.block(
                    [
                        [
                            R.from_euler("xyz", [0, 1.2, 2.3], degrees=True).as_matrix(),
                            np.array([1.2, 2.5, 3.4]).reshape((3, 1)),
                        ],
                        [np.array([0, 0, 0, 1])],
                    ]
                ),
            ]
        )
        T_rig_world_timestamps_us = np.linspace(
            ref_sequence_timestamp_interval_us.start,
            ref_sequence_timestamp_interval_us.stop - 1,
            num=len(T_rig_worlds),
            dtype=np.uint64,
        )

        coverate_pose_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            "throwaway_poses_type",
            group_name=None,  # use default component group
        )
        with self.assertRaises(AssertionError):
            coverate_pose_writer.store_dynamic_pose(
                source_frame_id="rig",
                target_frame_id="world",
                poses=T_rig_worlds[: len(T_rig_worlds) - 1],
                timestamps_us=T_rig_world_timestamps_us[: len(T_rig_worlds) - 1],  # insufficient coverage
            )
        coverate_pose_writer.store_dynamic_pose(
            source_frame_id="some",
            target_frame_id="coordinate",
            poses=T_rig_worlds[: len(T_rig_worlds) - 1],
            timestamps_us=T_rig_world_timestamps_us[: len(T_rig_worlds) - 1],
            require_sequence_time_coverage=False,
        )
        with self.assertRaises(AssertionError):
            coverate_pose_writer.store_dynamic_pose(
                source_frame_id="rig",
                target_frame_id="world",
                poses=T_rig_worlds[1:],
                timestamps_us=T_rig_world_timestamps_us[1:],  # insufficient coverage
            )
        coverate_pose_writer.store_dynamic_pose(
            source_frame_id="other",
            target_frame_id="frame",
            poses=T_rig_worlds[1:],
            timestamps_us=T_rig_world_timestamps_us[1:],
            require_sequence_time_coverage=False,
        )

        store_writer.register_component_writer(
            PosesComponent.Writer,
            ref_poses_id := "some_poses_type",
            group_name=None,  # use default component group
            generic_meta_data=cast(Dict[str, JsonLike], ref_pose_generic_meta_data := {"some": "thing"}),
        ).store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=(ref_T_rig_worlds := T_rig_worlds),
            timestamps_us=(ref_T_rig_world_timestamps_us := T_rig_world_timestamps_us),
        ).store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=(ref_T_world_world_global := T_world_world_global),
        ).store_static_pose(
            source_frame_id=(ref_camera_id := "ref_camera_id"),
            target_frame_id="rig",
            pose=(
                ref_camera_T_sensor_rig := np.block(
                    [
                        [
                            R.from_euler("xyz", [1, 1, 3], degrees=True).as_matrix(),
                            np.array([2, 1, -1]).reshape((3, 1)),
                        ],
                        [np.array([0, 0, 0, 1])],
                    ],
                ).astype(np.float32)
            ),
        ).store_static_pose(
            source_frame_id=(ref_lidar_id := "some-lidar-sensor-name"),
            target_frame_id="rig",
            pose=(
                ref_lidar_T_sensor_rig := np.block(
                    [
                        [
                            R.from_euler("xyz", [2, 1, 3], degrees=True).as_matrix(),
                            np.array([3, 1, -1]).reshape((3, 1)),
                        ],
                        [np.array([0, 0, 0, 1])],
                    ]
                ).astype(np.float32)
            ),
        ).store_static_pose(
            source_frame_id=(ref_radar_id := "some-radar-sensor-name"),
            target_frame_id="rig",
            pose=(
                ref_radar_T_sensor_rig := np.block(
                    [
                        [
                            R.from_euler("xyz", [2, 2, 3], degrees=True).as_matrix(),
                            np.array([3, 2, -1]).reshape((3, 1)),
                        ],
                        [np.array([0, 0, 0, 1])],
                    ]
                ).astype(np.float32)
            ),
        )

        # Store intrinsics
        intrinsic_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer, ref_intrinsics_id := "default", "intrinsics"
        )

        intrinsic_writer.store_camera_intrinsics(
            ref_camera_id,
            ref_camera_intrinsics := OpenCVFisheyeCameraModelParameters(
                resolution=np.array([3840, 2160], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
                focal_length=np.array(
                    [
                        1913.76478,
                        1913.99708,
                    ],
                    dtype=np.float32,
                ),
                radial_coeffs=np.array(
                    [
                        -0.030093122,
                        -0.005103817,
                        -0.000849622,
                        0.001079542,
                    ],
                    dtype=np.float32,
                ),
                max_angle=np.deg2rad(140 / 2),
                external_distortion_parameters=BivariateWindshieldModelParameters(
                    reference_poly=ReferencePolynomial.FORWARD,
                    horizontal_poly=np.array(
                        [
                            -0.000475919834570959,
                            0.99944007396698,
                            0.000166745347087272,
                            0.000205887947231531,
                            0.0055195577442646,
                            0.000861024134792387,
                        ],
                        dtype=np.float32,
                    ),
                    vertical_poly=np.array(
                        [
                            0.00152770057320595,
                            -0.000532537756953388,
                            -5.65027039556298e-05,
                            -4.02410341848736e-06,
                            0.000608163303695619,
                            1.0094313621521,
                            -0.00125278066843748,
                            0.00823396816849709,
                            -0.000293767458060756,
                            0.0185473654419184,
                            -0.003074218519032,
                            0.00599765172228217,
                            0.0172030478715897,
                            -0.00364979170262814,
                            0.0069147446192801,
                        ],
                        dtype=np.float32,
                    ),
                    horizontal_poly_inverse=np.array(
                        [
                            0.0004770369,
                            1.0005774,
                            -0.00016896478,
                            -0.00020207358,
                            -0.0054899976,
                            -0.0008536868,
                        ],
                        dtype=np.float32,
                    ),
                    vertical_poly_inverse=np.array(
                        [
                            -0.0015191488,
                            0.00052959577,
                            7.882431e-05,
                            -6.966009e-06,
                            -0.00059701066,
                            0.9906775,
                            0.00116782,
                            -0.007893825,
                            0.00026140467,
                            -0.017767625,
                            0.0027627628,
                            -0.00544897,
                            -0.015480865,
                            0.0033684247,
                            -0.0057964055,
                        ],
                        dtype=np.float32,
                    ),
                ),
            ),
        )

        intrinsic_writer.store_lidar_intrinsics(
            ref_lidar_id,
            ref_lidar_intrinsics := RowOffsetStructuredSpinningLidarModelParameters(
                spinning_frequency_hz=10.0,
                spinning_direction="ccw",
                n_rows=128,
                n_columns=3600,
                row_elevations_rad=np.linspace(0.2511354088783264, -0.4364195466041565, 128, dtype=np.float32),
                column_azimuths_rad=np.linspace(-3.141576051712036, 3.141592502593994, 3600, dtype=np.float32),
                row_azimuth_offsets_rad=np.linspace(0.0, 0.0, 128, dtype=np.float32),
            ),
        )

        # Store camera masks
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            ref_masks_id := "default",
            "masks",
            ref_masks_generic_meta_data := {"some-meta-data": np.random.default_rng().random((3, 2)).tolist()},
        )

        masks_writer.store_camera_masks(
            ref_camera_id,
            {
                (ref_camera_mask_name := "default"): (
                    ref_camera_mask_image := PILImage.fromarray(
                        np.random.default_rng().random((3840, 2160)) > 0.5
                    ).resize((480, 270))
                )
            },
        )

        # Store camera data
        camera_writer = store_writer.register_component_writer(
            CameraSensorComponent.Writer,
            ref_camera_id,
            "cameras",
            ref_camera_generic_meta_data := {"some-meta-data": np.random.default_rng().random((3, 2)).tolist()},
        )

        with io.BytesIO() as buffer:
            PILImage.fromarray(np.random.default_rng().integers(0, 256, (640, 480, 3), dtype=np.uint8)).save(
                buffer, format="png", optimize=True, quality=91
            )

            camera_writer.store_frame(
                ref_image_binary0 := buffer.getvalue(),
                "png",
                ref_camera_timestamps_us0 := np.array([0 * 1e6, 0.1 * 1e6], dtype=np.uint64),
                ref_camera_generic_data0 := {"some-frame-data": np.random.default_rng().random((6, 2))},
                ref_camera_generic_metadata0 := cast(
                    Dict[str, JsonLike], {"some-frame-meta-data": {"something": 1, "else": 2}}
                ),
            )

        with io.BytesIO() as buffer:
            PILImage.fromarray(np.random.default_rng().integers(0, 256, (640, 480, 3), dtype=np.uint8)).save(
                buffer, format="png", optimize=True, quality=91
            )

            camera_writer.store_frame(
                ref_image_binary1 := buffer.getvalue(),
                "png",
                ref_camera_timestamps_us1 := np.array([0.1 * 1e6, 0.2 * 1e6], dtype=np.uint64),
                ref_camera_generic_data1 := {"some-frame-data": np.random.default_rng().random((6, 2))},
                ref_camera_generic_metadata1 := cast(
                    Dict[str, JsonLike], {"some-more-frame-meta-data": {"even": True, "more": None}}
                ),
            )

        # Store lidar data
        lidar_writer = store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            ref_lidar_id,
            "lidars",
            ref_lidar_generic_meta_data := {"some-lidar-meta-data": np.random.default_rng().random((3, 2)).tolist()},
        )

        def normalize_points(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            norms = np.linalg.norm(vectors, axis=1)
            return vectors / norms[:, np.newaxis], norms

        ref_lidar_direction0, lidar_distance_m0 = normalize_points(
            np.random.default_rng().random((5, 3)).astype(np.float32) + 0.1
        )

        lidar_writer.store_frame(
            ref_lidar_direction0,
            ref_lidar_timestamp_us0 := np.linspace(0 * 1e6, 0.5 * 1e6, num=5, dtype=np.uint64),
            ref_lidar_model_element0 := np.arange(5 * 2, dtype=np.uint16).reshape((5, 2)),
            ref_lidar_distance_m0 := lidar_distance_m0[np.newaxis, :],
            ref_lidar_intensity0 := np.random.default_rng().random((1, 5)).astype(np.float32),
            ref_lidar_timestamps_us0 := np.array([0 * 1e6, 0.5 * 1e6], dtype=np.uint64),
            ref_lidar_generic_data0 := {"some-other-frame-data": np.random.default_rng().random((6, 2))},
            ref_lidar_generic_metadata0 := cast(
                Dict[str, JsonLike], {"some-more-meta-data": {"yes": None, "no": True}}
            ),
        )

        ref_lidar_valid_mask0 = np.ones((1, 5), dtype=bool)

        ref_lidar_direction_1, lidar_distance_m1 = normalize_points(
            np.random.default_rng().random((8, 3)).astype(np.float32) + 0.1
        )

        abscent_mask = np.stack(
            # first return all valid
            (
                np.zeros((8), dtype=bool),
                # some of the second return consistently invalid
                np.random.default_rng().random((8)) > 0.25,
            )
        )
        ref_lidar_distance_m1 = np.stack((lidar_distance_m1, lidar_distance_m1 + 0.1))
        ref_lidar_distance_m1[abscent_mask] = np.nan
        ref_lidar_intensity1 = np.random.default_rng().random((2, 8)).astype(np.float32)
        ref_lidar_intensity1[abscent_mask] = np.nan

        ref_lidar_valid_mask1 = ~abscent_mask

        lidar_writer.store_frame(
            ref_lidar_direction_1,
            ref_lidar_timestamp_us1 := np.linspace(0.5 * 1e6, 1 * 1e6, num=8, dtype=np.uint64),
            None,
            ref_lidar_distance_m1,
            ref_lidar_intensity1,
            ref_lidar_timestamps_us1 := np.array([0.5 * 1e6, 1 * 1e6], dtype=np.uint64),
            ref_lidar_generic_data1 := {"some-other-frame-data": np.random.default_rng().random((2, 2))},
            ref_lidar_generic_metadata1 := cast(Dict[str, JsonLike], {"even-more-meta-data": {"yesno": None}}),
        )

        # Store radar data
        radar_writer = store_writer.register_component_writer(
            RadarSensorComponent.Writer,
            ref_radar_id,
            "special-radars",
            ref_radar_generic_meta_data := {"some-radar-meta-data": np.random.default_rng().random((3, 2)).tolist()},
        )

        ref_radar_direction_m0, radar_distance_m0 = normalize_points(
            np.random.default_rng().random((5, 3)).astype(np.float32) + 0.2
        )

        radar_writer.store_frame(
            ref_radar_direction_m0,
            ref_radar_timestamp_us0 := np.array([0.1 * 1e6] * 5, dtype=np.uint64),
            ref_radar_distance_m0 := radar_distance_m0[np.newaxis, :],
            ref_radar_timestamps_us0 := np.array([0.1 * 1e6, 0.1 * 1e6], dtype=np.uint64),
            ref_radar_generic_data0 := {"some-other-frame-data": np.random.default_rng().random((6, 2))},
            ref_radar_generic_metadata0 := cast(
                Dict[str, JsonLike], {"some-more-meta-data": {"funny": "yes", "no": True}}
            ),
        )

        ref_radar_direction_m1, radar_distance_m1 = normalize_points(
            np.random.default_rng().random((8, 3)).astype(np.float32) + 0.2
        )

        radar_writer.store_frame(
            ref_radar_direction_m1,
            ref_radar_timestamp_us1 := np.array([0.2 * 1e6] * 8, dtype=np.uint64),
            ref_radar_distance_m1 := np.stack((radar_distance_m1, radar_distance_m1 + 0.2)),
            ref_radar_timestamps_us1 := np.array([0.2 * 1e6, 0.2 * 1e6], dtype=np.uint64),
            ref_radar_generic_data1 := {"some-radar-frame-data": np.random.default_rng().random((6, 2))},
            ref_radar_generic_metadata1 := cast(Dict[str, JsonLike], {"some-more-meta-data": {"funny": ":("}}),
        )

        ## Finalize writers up to here
        store_paths = store_writer.finalize()

        ## Simulated adding additional components by instantiating a new sequence writer from the existing meta-data
        store_writer = SequenceComponentGroupsWriter.from_reader(
            output_dir_path=store_writer._output_dir_path,
            store_base_name=store_writer._store_base_name,
            sequence_reader=SequenceComponentGroupsReader(store_paths, open_consolidated=open_consolidated),
            store_type=self.store_type,
        )

        # Store cuboids with the new writer
        cuboids_writer = store_writer.register_component_writer(
            CuboidsComponent.Writer,
            ref_cuboids_id := "default",
            "cuboids",
            ref_cuboids_generic_meta_data := cast(Dict[str, JsonLike], {"track-set-meta-data": "some-value"}),
        )

        ref_observation = CuboidTrackObservation(
            track_id="track-1",
            class_id="car",
            timestamp_us=int(0.3 * 1e6),
            reference_frame_timestamp_us=int(0.5 * 1e6),
            bbox3=BBox3(
                centroid=(1.0, 2.0, 3.0),
                dim=(4.0, 2.0, 1.5),
                rot=(0.0, 0.0, 0.0),
            ),
            reference_frame_id=ref_lidar_id,
            source=LabelSource.AUTOLABEL,
            source_version="v0",
        )

        with self.assertRaises(AssertionError):
            cuboids_writer.store_observations(
                cuboid_observations=[
                    dataclasses.replace(ref_observation, timestamp_us=ref_sequence_timestamp_interval_us.stop + 10)
                ]
            )

        with self.assertRaises(AssertionError):
            cuboids_writer.store_observations(
                cuboid_observations=[
                    dataclasses.replace(
                        ref_observation, reference_frame_timestamp_us=ref_sequence_timestamp_interval_us.stop + 10
                    )
                ]
            )

        cuboids_writer.store_observations(
            cuboid_observations=(
                ref_cuboid_observations := [
                    ref_observation,
                    CuboidTrackObservation(
                        track_id="track-1",
                        class_id="car",
                        timestamp_us=int(0.4 * 1e6),
                        reference_frame_timestamp_us=int(1.0 * 1e6),
                        reference_frame_id=ref_lidar_id,
                        source=LabelSource.AUTOLABEL,
                        source_version="v0",
                        bbox3=BBox3(
                            centroid=(1.5, 2.5, 3.5),
                            dim=(4.0, 2.0, 1.5),
                            rot=(0.0, 0.0, 0.1),
                        ),
                    ),
                ]
            ),
        )

        ## Finalize additional writers
        store_paths += store_writer.finalize()

        ## Reload sequence and verify consistency
        store_reader = SequenceComponentGroupsReader(store_paths, open_consolidated=open_consolidated)

        # check sequence data
        self.assertEqual(store_reader.sequence_id, ref_sequence_id)
        self.assertEqual(store_reader.sequence_timestamp_interval_us, ref_sequence_timestamp_interval_us)
        self.assertEqual(store_reader.generic_meta_data, ref_generic_sequence_meta_data)

        # check rig pose / calibration data
        poses_readers = store_reader.open_component_readers(PosesComponent.Reader)

        self.assertEqual(len(poses_readers), 2)
        poses_reader = poses_readers[ref_poses_id]

        self.assertEqual(poses_reader.instance_name, ref_poses_id)
        self.assertEqual(poses_reader.generic_meta_data, ref_pose_generic_meta_data)

        self.assertTrue(np.all(poses_reader.get_static_pose("world", "world_global") == ref_T_world_world_global))

        T_rig_worlds, T_rig_world_timestamps_us = poses_reader.get_dynamic_pose("rig", "world")
        self.assertTrue(np.all(T_rig_worlds == ref_T_rig_worlds))
        self.assertTrue(np.all(T_rig_world_timestamps_us == ref_T_rig_world_timestamps_us))

        self.assertTrue(np.all(poses_reader.get_static_pose(ref_camera_id, "rig") == ref_camera_T_sensor_rig))
        self.assertTrue(np.all(poses_reader.get_static_pose(ref_lidar_id, "rig") == ref_lidar_T_sensor_rig))
        self.assertTrue(np.all(poses_reader.get_static_pose(ref_radar_id, "rig") == ref_radar_T_sensor_rig))

        with self.assertRaises(KeyError):
            poses_reader.get_static_pose("non-existing-sensor", "rig")

        with self.assertRaises(KeyError):
            poses_reader.get_dynamic_pose("non-existing-frame", "world")

        all_static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("world", "world_global"), all_static_poses)
        self.assertTrue(np.all(all_static_poses[("world", "world_global")] == ref_T_world_world_global))
        self.assertIn((ref_camera_id, "rig"), all_static_poses)
        self.assertTrue(np.all(all_static_poses[(ref_camera_id, "rig")] == ref_camera_T_sensor_rig))
        self.assertIn((ref_lidar_id, "rig"), all_static_poses)
        self.assertTrue(np.all(all_static_poses[(ref_lidar_id, "rig")] == ref_lidar_T_sensor_rig))
        self.assertIn((ref_radar_id, "rig"), all_static_poses)
        self.assertTrue(np.all(all_static_poses[(ref_radar_id, "rig")] == ref_radar_T_sensor_rig))

        all_dynamic_poses = dict(poses_reader.get_dynamic_poses())
        self.assertIn(("rig", "world"), all_dynamic_poses)
        dyn_poses, dyn_timestamps = all_dynamic_poses[("rig", "world")]
        self.assertTrue(np.all(dyn_poses == ref_T_rig_worlds))
        self.assertTrue(np.all(dyn_timestamps == ref_T_rig_world_timestamps_us))

        # check intrinsics data
        intrinsic_readers = store_reader.open_component_readers(IntrinsicsComponent.Reader)

        self.assertEqual(len(intrinsic_readers), 1)
        intrinsic_reader = intrinsic_readers[ref_intrinsics_id]

        self.assertEqual(
            (camera_model_parameters := intrinsic_reader.get_camera_model_parameters(ref_camera_id)).to_dict(),
            ref_camera_intrinsics.to_dict(),
        )
        self.assertIsInstance(camera_model_parameters, OpenCVFisheyeCameraModelParameters)
        self.assertIsInstance(
            camera_model_parameters.external_distortion_parameters, BivariateWindshieldModelParameters
        )

        self.assertEqual(
            (
                lidar_model_parameters := unpack_optional(intrinsic_reader.get_lidar_model_parameters(ref_lidar_id))
            ).to_dict(),
            ref_lidar_intrinsics.to_dict(),
        )
        self.assertIsInstance(lidar_model_parameters, RowOffsetStructuredSpinningLidarModelParameters)

        # check masks data
        masks_readers = store_reader.open_component_readers(MasksComponent.Reader)

        self.assertEqual(len(masks_readers), 1)
        masks_reader = masks_readers[ref_masks_id]

        self.assertEqual(masks_reader.instance_name, ref_masks_id)
        self.assertEqual(masks_reader.generic_meta_data, ref_masks_generic_meta_data)

        self.assertEqual(masks_reader.get_camera_mask_names(ref_camera_id), [ref_camera_mask_name])
        self.assertEqual(
            masks_reader.get_camera_mask_image(ref_camera_id, ref_camera_mask_name).tobytes(),
            ref_camera_mask_image.tobytes(),
        )
        for mask_name, mask_image in masks_reader.get_camera_mask_images(ref_camera_id):
            self.assertEqual(mask_name, ref_camera_mask_name)
            self.assertEqual(mask_image.tobytes(), ref_camera_mask_image.tobytes())

        # check camera data
        camera_readers = store_reader.open_component_readers(CameraSensorComponent.Reader)

        self.assertEqual(len(camera_readers), 1)
        camera_reader = camera_readers[ref_camera_id]

        self.assertEqual(camera_reader.instance_name, ref_camera_id)
        self.assertEqual(camera_reader.generic_meta_data, ref_camera_generic_meta_data)

        self.assertTrue(
            np.all(
                camera_reader.frames_timestamps_us == np.stack([ref_camera_timestamps_us0, ref_camera_timestamps_us1])
            )
        )

        with self.assertRaises(KeyError):
            camera_reader.get_frame_timestamps_us(1234)

        self.assertTrue(
            np.all(camera_reader.get_frame_timestamps_us(ref_camera_timestamps_us0[1]) == ref_camera_timestamps_us0)
        )
        self.assertTrue(
            np.all(camera_reader.get_frame_timestamps_us(ref_camera_timestamps_us1[1]) == ref_camera_timestamps_us1)
        )

        self.assertEqual(
            (
                frame0_data := camera_reader.get_frame_handle(ref_camera_timestamps_us0[1]).get_data()
            ).get_encoded_image_data(),
            ref_image_binary0,
        )
        self.assertEqual(frame0_data.get_encoded_image_format(), "png")
        self.assertEqual(
            (
                frame1_data := camera_reader.get_frame_handle(ref_camera_timestamps_us1[1]).get_data()
            ).get_encoded_image_data(),
            ref_image_binary1,
        )
        self.assertEqual(frame1_data.get_encoded_image_format(), "png")

        self.assertEqual(
            names := camera_reader.get_frame_generic_data_names(ref_camera_timestamps_us0[1]),
            list(ref_camera_generic_data0.keys()),
        )
        for name in names:
            np.testing.assert_array_equal(
                camera_reader.get_frame_generic_data(ref_camera_timestamps_us0[1], name),
                ref_camera_generic_data0[name],
            )
        self.assertEqual(
            names := camera_reader.get_frame_generic_data_names(ref_camera_timestamps_us1[1]),
            list(ref_camera_generic_data1.keys()),
        )
        for name in names:
            np.testing.assert_array_equal(
                camera_reader.get_frame_generic_data(ref_camera_timestamps_us1[1], name),
                ref_camera_generic_data1[name],
            )

        self.assertEqual(
            camera_reader.get_frame_generic_meta_data(ref_camera_timestamps_us0[1]), ref_camera_generic_metadata0
        )
        self.assertEqual(
            camera_reader.get_frame_generic_meta_data(ref_camera_timestamps_us1[1]), ref_camera_generic_metadata1
        )

        # check lidar data
        lidar_readers = store_reader.open_component_readers(LidarSensorComponent.Reader)

        self.assertEqual(len(lidar_readers), 1)
        lidar_reader = lidar_readers[ref_lidar_id]

        self.assertEqual(lidar_reader.instance_name, ref_lidar_id)
        self.assertEqual(lidar_reader.generic_meta_data, ref_lidar_generic_meta_data)

        self.assertTrue(
            np.all(lidar_reader.frames_timestamps_us == np.stack([ref_lidar_timestamps_us0, ref_lidar_timestamps_us1]))
        )

        self.assertTrue(
            np.all(lidar_reader.get_frame_timestamps_us(ref_lidar_timestamps_us0[1]) == ref_lidar_timestamps_us0)
        )
        self.assertTrue(
            np.all(lidar_reader.get_frame_timestamps_us(ref_lidar_timestamps_us1[1]) == ref_lidar_timestamps_us1)
        )

        self.assertEqual(lidar_reader.get_frame_ray_bundle_count(ref_lidar_timestamps_us0[1]), 5)
        self.assertEqual(lidar_reader.get_frame_ray_bundle_count(ref_lidar_timestamps_us1[1]), 8)
        ref_ray_bundle_data_names = [
            "direction",
            "timestamp_us",
        ]
        self.assertEqual(
            set(lidar_reader.get_frame_ray_bundle_data_names(ref_lidar_timestamps_us0[1])),
            set(ref_ray_bundle_data_names + ["model_element"]),
        )
        self.assertEqual(
            set(lidar_reader.get_frame_ray_bundle_data_names(ref_lidar_timestamps_us1[1])),
            set(ref_ray_bundle_data_names),
        )
        for name in ref_ray_bundle_data_names + ["model_element"]:
            self.assertTrue(lidar_reader.has_frame_ray_bundle_data(ref_lidar_timestamps_us0[1], name))

        for name in ref_ray_bundle_data_names:
            self.assertTrue(lidar_reader.has_frame_ray_bundle_data(ref_lidar_timestamps_us1[1], name))

        self.assertEqual(lidar_reader.get_frame_ray_bundle_return_count(ref_lidar_timestamps_us0[1]), 1)
        self.assertEqual(lidar_reader.get_frame_ray_bundle_return_count(ref_lidar_timestamps_us1[1]), 2)
        ref_ray_bundle_returns_data_names = [
            "distance_m",
            "intensity",
        ]
        self.assertEqual(
            set(lidar_reader.get_frame_ray_bundle_return_data_names(ref_lidar_timestamps_us0[1])),
            set(ref_ray_bundle_returns_data_names),
        )
        self.assertEqual(
            set(lidar_reader.get_frame_ray_bundle_return_data_names(ref_lidar_timestamps_us1[1])),
            set(ref_ray_bundle_returns_data_names),
        )
        for name in ref_ray_bundle_returns_data_names:
            self.assertTrue(lidar_reader.has_frame_ray_bundle_return_data(ref_lidar_timestamps_us0[1], name))
            self.assertTrue(lidar_reader.has_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], name))

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_data(ref_lidar_timestamps_us0[1], "direction") == ref_lidar_direction0
            )
        )
        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_data(ref_lidar_timestamps_us1[1], "direction")
                == ref_lidar_direction_1
            )
        )

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_data(ref_lidar_timestamps_us0[1], "timestamp_us")
                == ref_lidar_timestamp_us0
            )
        )
        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_data(ref_lidar_timestamps_us1[1], "timestamp_us")
                == ref_lidar_timestamp_us1
            )
        )

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_data(ref_lidar_timestamps_us0[1], "model_element")
                == ref_lidar_model_element0
            )
        )

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us0[1], "distance_m", None)
                == ref_lidar_distance_m0
            )
        )
        self.assertTrue(
            np.all(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us0[1], "distance_m", 0)
                == ref_lidar_distance_m0[0]
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], "distance_m", None),
                ref_lidar_distance_m1,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], "distance_m", 0),
                ref_lidar_distance_m1[0],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], "distance_m", 1),
                ref_lidar_distance_m1[1],
                equal_nan=True,
            )
        )

        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us0[1], "intensity", None),
                ref_lidar_intensity0,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us0[1], "intensity", 0),
                ref_lidar_intensity0[0],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], "intensity", None),
                ref_lidar_intensity1,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], "intensity", 0),
                ref_lidar_intensity1[0],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_data(ref_lidar_timestamps_us1[1], "intensity", 1),
                ref_lidar_intensity1[1],
                equal_nan=True,
            )
        )

        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_valid_mask(ref_lidar_timestamps_us0[1]), ref_lidar_valid_mask0
            )
        )

        self.assertTrue(
            np.array_equal(
                lidar_reader.get_frame_ray_bundle_return_valid_mask(ref_lidar_timestamps_us1[1]), ref_lidar_valid_mask1
            )
        )

        self.assertEqual(
            names := lidar_reader.get_frame_generic_data_names(ref_lidar_timestamps_us0[1]),
            list(ref_lidar_generic_data0.keys()),
        )
        for name in names:
            np.testing.assert_array_equal(
                lidar_reader.get_frame_generic_data(ref_lidar_timestamps_us0[1], name),
                ref_lidar_generic_data0[name],
            )
        self.assertEqual(
            names := lidar_reader.get_frame_generic_data_names(ref_lidar_timestamps_us1[1]),
            list(ref_lidar_generic_data1.keys()),
        )
        for name in names:
            np.testing.assert_array_equal(
                lidar_reader.get_frame_generic_data(ref_lidar_timestamps_us1[1], name),
                ref_lidar_generic_data1[name],
            )

        self.assertEqual(
            lidar_reader.get_frame_generic_meta_data(ref_lidar_timestamps_us0[1]), ref_lidar_generic_metadata0
        )
        self.assertEqual(
            lidar_reader.get_frame_generic_meta_data(ref_lidar_timestamps_us1[1]), ref_lidar_generic_metadata1
        )

        # check radar data
        radar_readers = store_reader.open_component_readers(RadarSensorComponent.Reader)

        self.assertEqual(len(radar_readers), 1)
        radar_reader = radar_readers[ref_radar_id]

        self.assertEqual(radar_reader.instance_name, ref_radar_id)
        self.assertEqual(radar_reader.generic_meta_data, ref_radar_generic_meta_data)

        self.assertTrue(
            np.all(radar_reader.frames_timestamps_us == np.stack([ref_radar_timestamps_us0, ref_radar_timestamps_us1]))
        )

        self.assertTrue(
            np.all(radar_reader.get_frame_timestamps_us(ref_radar_timestamps_us0[1]) == ref_radar_timestamps_us0)
        )
        self.assertTrue(
            np.all(radar_reader.get_frame_timestamps_us(ref_radar_timestamps_us1[1]) == ref_radar_timestamps_us1)
        )

        self.assertEqual(radar_reader.get_frame_ray_bundle_count(ref_radar_timestamps_us0[1]), 5)
        self.assertEqual(radar_reader.get_frame_ray_bundle_count(ref_radar_timestamps_us1[1]), 8)
        ref_ray_bundle_data_names = [
            "direction",
            "timestamp_us",
        ]
        self.assertEqual(
            set(radar_reader.get_frame_ray_bundle_data_names(ref_radar_timestamps_us0[1])),
            set(ref_ray_bundle_data_names),
        )
        self.assertEqual(
            set(radar_reader.get_frame_ray_bundle_data_names(ref_radar_timestamps_us1[1])),
            set(ref_ray_bundle_data_names),
        )
        for name in ref_ray_bundle_data_names:
            self.assertTrue(radar_reader.has_frame_ray_bundle_data(ref_radar_timestamps_us0[1], name))
            self.assertTrue(radar_reader.has_frame_ray_bundle_data(ref_radar_timestamps_us1[1], name))

        self.assertEqual(radar_reader.get_frame_ray_bundle_return_count(ref_radar_timestamps_us0[1]), 1)
        self.assertEqual(radar_reader.get_frame_ray_bundle_return_count(ref_radar_timestamps_us1[1]), 2)
        ref_ray_bundle_returns_data_names = [
            "distance_m",
        ]
        self.assertEqual(
            set(radar_reader.get_frame_ray_bundle_return_data_names(ref_radar_timestamps_us0[1])),
            set(ref_ray_bundle_returns_data_names),
        )
        self.assertEqual(
            set(radar_reader.get_frame_ray_bundle_return_data_names(ref_radar_timestamps_us1[1])),
            set(ref_ray_bundle_returns_data_names),
        )
        for name in ref_ray_bundle_returns_data_names:
            self.assertTrue(radar_reader.has_frame_ray_bundle_return_data(ref_radar_timestamps_us0[1], name))
            self.assertTrue(radar_reader.has_frame_ray_bundle_return_data(ref_radar_timestamps_us1[1], name))

        self.assertTrue(
            np.all(
                radar_reader.get_frame_ray_bundle_data(ref_radar_timestamps_us0[1], "direction")
                == ref_radar_direction_m0
            )
        )
        self.assertTrue(
            np.all(
                radar_reader.get_frame_ray_bundle_data(ref_radar_timestamps_us1[1], "direction")
                == ref_radar_direction_m1
            )
        )

        self.assertTrue(
            np.all(
                radar_reader.get_frame_ray_bundle_data(ref_radar_timestamps_us0[1], "timestamp_us")
                == ref_radar_timestamp_us0
            )
        )
        self.assertTrue(
            np.all(
                radar_reader.get_frame_ray_bundle_data(ref_radar_timestamps_us1[1], "timestamp_us")
                == ref_radar_timestamp_us1
            )
        )

        self.assertTrue(
            np.array_equal(
                radar_reader.get_frame_ray_bundle_return_data(ref_radar_timestamps_us0[1], "distance_m", None),
                ref_radar_distance_m0,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                radar_reader.get_frame_ray_bundle_return_data(ref_radar_timestamps_us0[1], "distance_m", 0),
                ref_radar_distance_m0[0],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                radar_reader.get_frame_ray_bundle_return_data(ref_radar_timestamps_us1[1], "distance_m", None),
                ref_radar_distance_m1,
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                radar_reader.get_frame_ray_bundle_return_data(ref_radar_timestamps_us1[1], "distance_m", 0),
                ref_radar_distance_m1[0],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.array_equal(
                radar_reader.get_frame_ray_bundle_return_data(ref_radar_timestamps_us1[1], "distance_m", 1),
                ref_radar_distance_m1[1],
                equal_nan=True,
            )
        )

        self.assertEqual(
            names := radar_reader.get_frame_generic_data_names(ref_radar_timestamps_us0[1]),
            list(ref_radar_generic_data0.keys()),
        )
        for name in names:
            np.testing.assert_array_equal(
                radar_reader.get_frame_generic_data(ref_radar_timestamps_us0[1], name),
                ref_radar_generic_data0[name],
            )
        self.assertEqual(
            names := radar_reader.get_frame_generic_data_names(ref_radar_timestamps_us1[1]),
            list(ref_radar_generic_data1.keys()),
        )
        for name in names:
            np.testing.assert_array_equal(
                radar_reader.get_frame_generic_data(ref_radar_timestamps_us1[1], name),
                ref_radar_generic_data1[name],
            )

        self.assertEqual(
            radar_reader.get_frame_generic_meta_data(ref_radar_timestamps_us0[1]), ref_radar_generic_metadata0
        )
        self.assertEqual(
            radar_reader.get_frame_generic_meta_data(ref_radar_timestamps_us1[1]), ref_radar_generic_metadata1
        )

        # check cuboid data
        cuboid_readers = store_reader.open_component_readers(CuboidsComponent.Reader)

        self.assertEqual(len(cuboid_readers), 1)
        cuboid_reader = cuboid_readers[ref_cuboids_id]
        self.assertEqual(cuboid_reader.instance_name, ref_cuboids_id)
        self.assertEqual(cuboid_reader.generic_meta_data, ref_cuboids_generic_meta_data)

        self.assertEqual(list(cuboid_reader.get_observations()), ref_cuboid_observations)


@parameterized_class(
    ("store_type"),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestDataNewComponent(unittest.TestCase):
    """
    Test to demonstrate how to extend an existing dataset with a new custom component.

    This serves as a reference example for users who want to:
    1. Create a custom component with reader/writer classes
    2. Extend an existing dataset by adding new component data
    3. Handle component versioning correctly
    """

    store_type: Literal["itar", "directory"]

    def setUp(self):
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

    def test_new_component_extension(self):
        """
        Complete example of extending a dataset with a custom component.

        Steps demonstrated:
        1. Create initial dataset with basic pose data
        2. Define a custom component (VelocityComponent) with reader/writer
        3. Extend the dataset using SequenceComponentGroupsWriter.from_reader()
        4. Verify the extended dataset can be read correctly
        [test-only: 5. Test version compatibility (reader handling old/new versions)]
        """

        tempdir = tempfile.TemporaryDirectory()

        # ============================================================================
        # STEP 1: Create an initial dataset with just some static pose reference data
        # ============================================================================

        # Step 1: Create an initial dataset with just some static pose reference data

        initial_store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(tempdir.name),
            store_base_name=(sequence_id := "test-sequence"),
            sequence_id=sequence_id,
            sequence_timestamp_interval_us=(timestamp_interval := HalfClosedInterval(int(0 * 1e6), int(10 * 1e6) + 1)),
            store_type=self.store_type,
            generic_meta_data={"dataset": "test", "version": 1.0},
        )

        # Store a simple static pose (sensor to rig transformation)
        ref_T_sensor_rig = np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        initial_store_writer.register_component_writer(
            PosesComponent.Writer,
            instance_name := "default_poses",
            group_name=None,
            generic_meta_data={"description": "Basic pose data"},
        ).store_static_pose(
            source_frame_id="sensor",
            target_frame_id="rig",
            pose=ref_T_sensor_rig,
        )

        # Finalize the initial dataset
        initial_store_paths = initial_store_writer.finalize()
        # Initial dataset created

        # ============================================================================
        # STEP 2: Define a custom component with reader and writer classes
        # ============================================================================

        # Step 2: Define a custom component with reader and writer classes

        # This is a simple example component that stores velocity data over time
        class VelocityComponent:
            """Custom component for storing velocity vectors over time"""

            COMPONENT_NAME: str = "com.myorg.velocity"

            class Writer(ComponentWriter):
                """Writer for velocity data - version v1"""

                @staticmethod
                def get_component_name() -> str:
                    return VelocityComponent.COMPONENT_NAME

                @staticmethod
                def get_component_version() -> str:
                    return "v1"  # This is version 1 of our component

                def __init__(self, component_group, sequence_timestamp_interval_us):
                    super().__init__(component_group, sequence_timestamp_interval_us)
                    self.velocities = []
                    self.timestamps = []

                def store_velocity(
                    self,
                    velocity: np.ndarray,  # 3D velocity vector [vx, vy, vz]
                    timestamp_us: int,
                ):
                    """Store a velocity measurement at a specific timestamp"""
                    assert velocity.shape == (3,), "Velocity must be a 3D vector"
                    assert np.issubdtype(velocity.dtype, np.floating), "Velocity must be float type"

                    self.velocities.append(velocity)
                    self.timestamps.append(timestamp_us)
                    return self

                def finalize(self):
                    """Write all velocity data to zarr storage"""
                    if self.velocities:
                        velocities_array = np.stack(self.velocities)
                        timestamps_array = np.array(self.timestamps, dtype=np.uint64)

                        # Store as zarr arrays
                        self._group.create_dataset(
                            "velocities",
                            data=velocities_array,
                            dtype=velocities_array.dtype,
                        )
                        self._group.create_dataset(
                            "timestamps_us",
                            data=timestamps_array,
                            dtype=np.uint64,
                        )

            class Reader(ComponentReader):
                """Reader for velocity data - supports v1"""

                @staticmethod
                def get_component_name() -> str:
                    return VelocityComponent.COMPONENT_NAME

                @staticmethod
                def supports_component_version(version: str) -> bool:
                    # This reader only supports v1
                    return version == "v1"

                def get_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
                    """Returns (velocities, timestamps_us) arrays"""
                    velocities = np.array(self._group["velocities"][:])
                    timestamps_us = np.array(self._group["timestamps_us"][:])
                    return velocities, timestamps_us

        # ============================================================================
        # STEP 3: Extend the existing dataset with the new custom component
        # ============================================================================

        # Step 3: Extend the existing dataset with the new custom component

        # First, open a reader to the initial dataset
        initial_reader = SequenceComponentGroupsReader(component_group_paths=initial_store_paths)

        # Verify we can read the initial data
        poses_readers = initial_reader.open_component_readers(PosesComponent.Reader)
        self.assertEqual(len(poses_readers), 1)
        self.assertIn(instance_name, poses_readers)

        # Create a new writer that extends the existing dataset IN PLACE
        # This is the KEY step: use from_reader() to create a writer with the same metadata
        # Note: Use the SAME output directory and base name to extend the existing dataset
        # The from_reader() method copies sequence metadata but NOT component data
        extended_store_writer = SequenceComponentGroupsWriter.from_reader(
            sequence_reader=initial_reader,
            output_dir_path=UPath(tempdir.name),  # Same directory as initial
            store_base_name=sequence_id + "-extension",
            store_type=self.store_type,
        )

        # Now add our custom velocity component to the extended dataset
        ref_velocities = np.array(
            [
                [1.0, 0.0, 0.0],  # Moving in +x direction
                [1.5, 0.5, 0.0],  # Accelerating, slight +y
                [2.0, 1.0, 0.1],  # Continuing acceleration
                [2.0, 1.0, 0.0],  # Constant velocity
                [1.5, 0.5, 0.0],  # Decelerating
            ],
            dtype=np.float32,
        )

        ref_velocity_timestamps = np.array(
            [
                int(0 * 1e6),
                int(2 * 1e6),
                int(4 * 1e6),
                int(6 * 1e6),
                int(8 * 1e6),
            ],
            dtype=np.uint64,
        )

        velocity_writer = extended_store_writer.register_component_writer(
            VelocityComponent.Writer,
            velocity_instance_name := "ego_velocity",
            group_name="velocity",  # Optional: organize in a subgroup
            generic_meta_data={"units": "m/s", "reference_frame": "rig"},
        )

        # Store all velocity measurements
        for vel, ts in zip(ref_velocities, ref_velocity_timestamps):
            velocity_writer.store_velocity(vel, ts)

        # Finalize the extended dataset - this adds new component paths
        extended_store_paths = initial_store_paths + extended_store_writer.finalize()
        # Extended dataset with velocity component created

        # ============================================================================
        # STEP 4: Verify we can reload the extended dataset correctly
        # ============================================================================

        # Step 4: Verify we can reload the extended dataset correctly

        # Open a reader for the extended dataset
        extended_reader = SequenceComponentGroupsReader(component_group_paths=extended_store_paths)

        # Verify sequence metadata is preserved
        self.assertEqual(extended_reader.sequence_id, sequence_id)
        self.assertEqual(
            extended_reader.sequence_timestamp_interval_us,
            timestamp_interval,
        )

        # Verify original pose data is still present
        extended_poses_readers = extended_reader.open_component_readers(PosesComponent.Reader)
        self.assertEqual(len(extended_poses_readers), 1)
        self.assertIn(instance_name, extended_poses_readers)

        # Check the static pose is still there
        static_poses = list(extended_poses_readers[instance_name].get_static_poses())
        self.assertEqual(len(static_poses), 1)
        (src, tgt), pose = static_poses[0]
        self.assertEqual(src, "sensor")
        self.assertEqual(tgt, "rig")
        np.testing.assert_array_almost_equal(pose, ref_T_sensor_rig)

        # Verify our new velocity component is present and readable
        velocity_readers = extended_reader.open_component_readers(VelocityComponent.Reader)
        self.assertEqual(len(velocity_readers), 1)
        self.assertIn(velocity_instance_name, velocity_readers)

        velocity_reader = velocity_readers[velocity_instance_name]
        self.assertEqual(velocity_reader.instance_name, velocity_instance_name)
        self.assertEqual(velocity_reader.component_version, "v1")
        self.assertEqual(
            velocity_reader.generic_meta_data,
            {"units": "m/s", "reference_frame": "rig"},
        )

        # Read and verify velocity data
        loaded_velocities, loaded_timestamps = velocity_reader.get_velocities()
        np.testing.assert_array_almost_equal(loaded_velocities, ref_velocities)
        np.testing.assert_array_equal(loaded_timestamps, ref_velocity_timestamps)

        # Extended dataset verified successfully

        # ============================================================================
        # STEP 5 (Optional): Test component version compatibility
        # ============================================================================

        # Step 5 (Optional): Test component version compatibility

        # Define a v2 writer with additional features (acceleration data)
        class VelocityComponentV2:
            """Version 2 with acceleration data"""

            COMPONENT_NAME: str = "com.myorg.velocity"

            class Writer(ComponentWriter):
                @staticmethod
                def get_component_name() -> str:
                    return VelocityComponentV2.COMPONENT_NAME

                @staticmethod
                def get_component_version() -> str:
                    return "v2"  # New version

                def __init__(self, component_group, sequence_timestamp_interval_us):
                    super().__init__(component_group, sequence_timestamp_interval_us)
                    self.velocities = []
                    self.accelerations = []  # NEW: acceleration data
                    self.timestamps = []

                def store_velocity_with_acceleration(
                    self,
                    velocity: np.ndarray,
                    acceleration: np.ndarray,  # NEW parameter
                    timestamp_us: int,
                ):
                    """Store velocity and acceleration at a timestamp"""
                    self.velocities.append(velocity)
                    self.accelerations.append(acceleration)
                    self.timestamps.append(timestamp_us)
                    return self

                def finalize(self):
                    if self.velocities:
                        velocities_array = np.stack(self.velocities)
                        accelerations_array = np.stack(self.accelerations)
                        timestamps_array = np.array(self.timestamps, dtype=np.uint64)

                        self._group.create_dataset("velocities", data=velocities_array)
                        self._group.create_dataset("accelerations", data=accelerations_array)  # NEW
                        self._group.create_dataset("timestamps_us", data=timestamps_array)

        # Create a backward-compatible reader that can read both v1 and v2
        class VelocityComponentBackwardCompatibleReader(ComponentReader):
            """Reader that supports both v1 and v2"""

            @staticmethod
            def get_component_name() -> str:
                return "com.myorg.velocity"

            @staticmethod
            def supports_component_version(version: str) -> bool:
                # This reader can handle v1 and v2
                return version in ["v1", "v2"]

            def get_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
                """Returns velocities (works for both v1 and v2)"""
                velocities = np.array(self._group["velocities"][:])
                timestamps_us = np.array(self._group["timestamps_us"][:])
                return velocities, timestamps_us

            def get_accelerations(self) -> np.ndarray:
                """Returns accelerations (only available in v2)"""
                if self.component_version == "v1":
                    raise ValueError("Acceleration data not available in v1")
                return np.array(self._group["accelerations"][:])

        # Test that v1 reader cannot read v2 data
        v2_store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(tempdir.name) / "v2",
            store_base_name="test_v2",
            sequence_id="test_v2",
            sequence_timestamp_interval_us=timestamp_interval,
            store_type=self.store_type,
            generic_meta_data={"version": "v2_test"},
        )

        ref_accelerations = np.array(
            [
                [0.5, 0.5, 0.1],
                [0.5, 0.5, -0.1],
                [0.0, 0.0, 0.0],
                [-0.5, -0.5, 0.0],
                [-0.5, -0.5, 0.0],
            ],
            dtype=np.float32,
        )

        v2_writer = v2_store_writer.register_component_writer(
            VelocityComponentV2.Writer,
            "velocity_v2",
            group_name="velocity",
        )

        for vel, acc, ts in zip(ref_velocities, ref_accelerations, ref_velocity_timestamps):
            v2_writer.store_velocity_with_acceleration(vel, acc, ts)

        v2_store_paths = v2_store_writer.finalize()

        # Try to open with v1 reader - should fail because it doesn't support v2
        v2_reader = SequenceComponentGroupsReader(component_group_paths=v2_store_paths)

        # This should return empty dict because v1 reader doesn't support v2
        v1_readers_for_v2 = v2_reader.open_component_readers(VelocityComponent.Reader)
        self.assertEqual(len(v1_readers_for_v2), 0, "v1 reader should not be able to read v2 components")
        # v1 reader correctly skips v2 components (returns empty dict)

        # But backward-compatible reader should work
        bc_readers = v2_reader.open_component_readers(VelocityComponentBackwardCompatibleReader)
        self.assertEqual(len(bc_readers), 1)
        bc_reader = bc_readers["velocity_v2"]

        # Can read velocities from v2
        loaded_vel, loaded_ts = bc_reader.get_velocities()
        np.testing.assert_array_almost_equal(loaded_vel, ref_velocities)

        # Can also read accelerations from v2
        loaded_acc = bc_reader.get_accelerations()
        np.testing.assert_array_almost_equal(loaded_acc, ref_accelerations)

        # Test that backward-compatible reader can also read v1 data
        bc_readers_v1 = extended_reader.open_component_readers(VelocityComponentBackwardCompatibleReader)
        self.assertEqual(len(bc_readers_v1), 1)
        bc_reader_v1 = bc_readers_v1[velocity_instance_name]

        # Can read velocities from v1
        loaded_vel_v1, _ = bc_reader_v1.get_velocities()
        np.testing.assert_array_almost_equal(loaded_vel_v1, ref_velocities)

        # But cannot read accelerations from v1
        with self.assertRaises(ValueError) as context:
            bc_reader_v1.get_accelerations()
        self.assertIn("not available in v1", str(context.exception))

        # Version compatibility tests passed - all tests completed successfully


@parameterized_class(
    ("store_type"),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestPointCloudsComponent(unittest.TestCase):
    """Round-trip tests for the PointCloudsComponent Writer/Reader."""

    store_type: Literal["itar", "directory"]

    def setUp(self):
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

    def _make_writer_reader(
        self, attribute_schemas={}
    ) -> Tuple[
        PointCloudsComponent.Writer, SequenceComponentGroupsWriter, tempfile.TemporaryDirectory, HalfClosedInterval
    ]:
        """Helper: create a SequenceComponentGroupsWriter, register a PointCloudsComponent.Writer,
        and return (writer, tempdir, timestamp_interval) so the caller can store PCs."""
        tmpdir = tempfile.TemporaryDirectory()
        timestamp_interval = HalfClosedInterval(0, 10_000_001)

        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(tmpdir.name),
            store_base_name=(seq_id := "pc-test-seq"),
            sequence_id=seq_id,
            sequence_timestamp_interval_us=timestamp_interval,
            store_type=self.store_type,
            generic_meta_data={},
        )

        pc_writer = store_writer.register_component_writer(
            PointCloudsComponent.Writer,
            "test_pc",
            coordinate_unit=PointCloud.CoordinateUnit.METERS,
            attribute_schemas=attribute_schemas,
        )
        return pc_writer, store_writer, tmpdir, timestamp_interval

    def _finalize_and_open_reader(self, store_writer: SequenceComponentGroupsWriter) -> PointCloudsComponent.Reader:
        """Finalize the writer, open a reader, and return the PointCloudsComponent.Reader."""
        store_paths = store_writer.finalize()
        reader = SequenceComponentGroupsReader(component_group_paths=store_paths)
        pc_readers = reader.open_component_readers(PointCloudsComponent.Reader)
        self.assertIn("test_pc", pc_readers)
        return pc_readers["test_pc"]

    def test_single_pc_with_attributes(self):
        """Write 1 PC with rgb (uint8, (N,3)) + normals (float32, (N,3)), read back, verify all fields."""
        schemas = {
            "rgb": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
                shape_suffix=(3,),
            ),
            "normal": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.DIRECTION,
                dtype=np.dtype("float32"),
                shape_suffix=(3,),
            ),
        }
        pc_writer, store_writer, tmpdir, _ = self._make_writer_reader(attribute_schemas=schemas)

        N = 100
        xyz = np.random.default_rng().random((N, 3)).astype(np.float32)
        rgb = np.random.default_rng().integers(0, 256, size=(N, 3), dtype=np.uint8)
        normals = np.random.default_rng().random((N, 3)).astype(np.float32)

        pc_writer.store_pc(
            xyz=xyz,
            reference_frame_id="world",
            reference_frame_timestamp_us=1_000_000,
            attributes={"rgb": rgb, "normal": normals},
        )

        reader = self._finalize_and_open_reader(store_writer)

        # Verify coordinate_unit
        self.assertEqual(reader.coordinate_unit, PointCloud.CoordinateUnit.METERS)

        # Verify counts
        self.assertEqual(reader.pcs_count, 1)
        np.testing.assert_array_equal(reader.pc_timestamps_us, np.array([1_000_000], dtype=np.uint64))

        # Verify attribute schema
        self.assertEqual(sorted(reader.attribute_names), ["normal", "rgb"])
        rgb_schema = reader.get_attribute_schema("rgb")
        self.assertEqual(rgb_schema.transform_type, PointCloud.AttributeTransformType.INVARIANT)
        self.assertEqual(rgb_schema.dtype, np.dtype("uint8"))
        self.assertEqual(rgb_schema.shape_suffix, (3,))

        normals_schema = reader.get_attribute_schema("normal")
        self.assertEqual(normals_schema.transform_type, PointCloud.AttributeTransformType.DIRECTION)
        self.assertEqual(normals_schema.dtype, np.dtype("float32"))
        self.assertEqual(normals_schema.shape_suffix, (3,))

        # Verify PC data
        np.testing.assert_array_almost_equal(reader.get_pc_xyz(0), xyz)
        np.testing.assert_array_equal(reader.get_pc_attribute(0, "rgb"), rgb)
        np.testing.assert_array_almost_equal(reader.get_pc_attribute(0, "normal"), normals)

        # Verify reference frame
        self.assertEqual(reader.get_pc_reference_frame_id(0), "world")
        self.assertEqual(reader.get_pc_reference_frame_timestamp_us(0), 1_000_000)

        tmpdir.cleanup()

    def test_multiple_pcs_different_ref_frames(self):
        """Write 2 PCs with different reference_frame_id, verify per-pc ref frames."""
        pc_writer, store_writer, tmpdir, _ = self._make_writer_reader()

        xyz1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        xyz2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)

        pc_writer.store_pc(
            xyz=xyz1,
            reference_frame_id="sensor_a",
            reference_frame_timestamp_us=100_000,
        )
        pc_writer.store_pc(
            xyz=xyz2,
            reference_frame_id="sensor_b",
            reference_frame_timestamp_us=200_000,
        )

        reader = self._finalize_and_open_reader(store_writer)

        self.assertEqual(reader.pcs_count, 2)
        np.testing.assert_array_equal(
            reader.pc_timestamps_us,
            np.array([100_000, 200_000], dtype=np.uint64),
        )

        # PC 0
        np.testing.assert_array_almost_equal(reader.get_pc_xyz(0), xyz1)
        self.assertEqual(reader.get_pc_reference_frame_id(0), "sensor_a")
        self.assertEqual(reader.get_pc_reference_frame_timestamp_us(0), 100_000)

        # PC 1
        np.testing.assert_array_almost_equal(reader.get_pc_xyz(1), xyz2)
        self.assertEqual(reader.get_pc_reference_frame_id(1), "sensor_b")
        self.assertEqual(reader.get_pc_reference_frame_timestamp_us(1), 200_000)

        tmpdir.cleanup()

    def test_attribute_schema_json_roundtrip(self):
        """AttributeSchema.to_dict() -> from_dict() preserves all fields."""
        original = PointCloudsComponent.AttributeSchema(
            transform_type=PointCloud.AttributeTransformType.DIRECTION,
            dtype=np.dtype("float64"),
            shape_suffix=(3,),
        )
        serialized = original.to_dict()

        # Verify serialized form uses strings (enum names are UPPERCASE)
        self.assertEqual(serialized["transform_type"], "DIRECTION")
        self.assertEqual(serialized["dtype"], "float64")
        self.assertEqual(serialized["shape_suffix"], [3])

        deserialized = PointCloudsComponent.AttributeSchema.from_dict(serialized)
        self.assertEqual(deserialized.transform_type, original.transform_type)
        self.assertEqual(deserialized.dtype, original.dtype)
        self.assertEqual(deserialized.shape_suffix, original.shape_suffix)

        # Also test scalar (empty shape_suffix)
        scalar_schema = PointCloudsComponent.AttributeSchema(
            transform_type=PointCloud.AttributeTransformType.INVARIANT,
            dtype=np.dtype("float32"),
            shape_suffix=(),
        )
        rt = PointCloudsComponent.AttributeSchema.from_dict(scalar_schema.to_dict())
        self.assertEqual(rt.shape_suffix, ())

    def test_writer_rejects_undeclared_attribute(self):
        """store_pc with attr not in schema -> AssertionError."""
        schemas = {
            "rgb": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
                shape_suffix=(3,),
            ),
        }
        pc_writer, _, tmpdir, _ = self._make_writer_reader(attribute_schemas=schemas)

        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        rgb = np.array([[128, 64, 32]], dtype=np.uint8)
        extra = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        with self.assertRaises(AssertionError):
            pc_writer.store_pc(
                xyz=xyz,
                reference_frame_id="world",
                reference_frame_timestamp_us=1_000_000,
                attributes={"rgb": rgb, "extra_attr": extra},
            )

        tmpdir.cleanup()

    def test_writer_rejects_missing_schema_attribute(self):
        """store_pc missing a schema attr -> AssertionError."""
        schemas = {
            "rgb": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
                shape_suffix=(3,),
            ),
            "normal": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.DIRECTION,
                dtype=np.dtype("float32"),
                shape_suffix=(3,),
            ),
        }
        pc_writer, _, tmpdir, _ = self._make_writer_reader(attribute_schemas=schemas)

        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        rgb = np.array([[128, 64, 32]], dtype=np.uint8)

        with self.assertRaises(AssertionError):
            pc_writer.store_pc(
                xyz=xyz,
                reference_frame_id="world",
                reference_frame_timestamp_us=1_000_000,
                attributes={"rgb": rgb},  # missing "normal"
            )

        tmpdir.cleanup()

    def test_writer_rejects_wrong_shape(self):
        """store_pc with wrong-shaped array -> AssertionError."""
        schemas = {
            "rgb": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
                shape_suffix=(3,),
            ),
        }
        pc_writer, _, tmpdir, _ = self._make_writer_reader(attribute_schemas=schemas)

        N = 10
        xyz = np.random.default_rng().random((N, 3), dtype=np.float32)
        # Wrong shape: (N, 4) instead of (N, 3)
        rgb_wrong = np.random.default_rng().integers(0, 256, size=(N, 4), dtype=np.uint8)

        with self.assertRaises(AssertionError):
            pc_writer.store_pc(
                xyz=xyz,
                reference_frame_id="world",
                reference_frame_timestamp_us=1_000_000,
                attributes={"rgb": rgb_wrong},
            )

        tmpdir.cleanup()

    def test_writer_rejects_reference_frame_timestamp_out_of_range(self):
        """store_pc with reference_frame_timestamp_us outside sequence range -> AssertionError."""
        pc_writer, _, tmpdir, _ = self._make_writer_reader()

        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            pc_writer.store_pc(
                xyz=xyz,
                reference_frame_id="world",
                reference_frame_timestamp_us=99_000_000,  # far outside sequence range
            )

        tmpdir.cleanup()

    def test_writer_rejects_wrong_xyz_dtype(self):
        """store_pc with float64 xyz raises AssertionError (float32 required)."""
        pc_writer, _, tmpdir, _ = self._make_writer_reader()

        xyz_f64 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        with self.assertRaises(AssertionError):
            pc_writer.store_pc(
                xyz=xyz_f64,  # type: ignore
                reference_frame_id="world",
                reference_frame_timestamp_us=500_000,
            )

        tmpdir.cleanup()

    def test_empty_writer_finalize(self):
        """Finalizing a writer with zero store_pc calls produces a valid empty reader."""
        _, store_writer, tmpdir, _ = self._make_writer_reader()

        # Finalize without any store_pc calls
        reader = self._finalize_and_open_reader(store_writer)

        self.assertEqual(reader.pcs_count, 0)
        np.testing.assert_array_equal(reader.pc_timestamps_us, np.array([], dtype=np.uint64))
        self.assertEqual(reader.attribute_names, [])

        tmpdir.cleanup()

    def test_no_attributes_no_generic_data(self):
        """Write/read a PC with empty schema and no generic data."""
        pc_writer, store_writer, tmpdir, _ = self._make_writer_reader()

        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc_writer.store_pc(
            xyz=xyz,
            reference_frame_id="ego",
            reference_frame_timestamp_us=500_000,
        )

        reader = self._finalize_and_open_reader(store_writer)

        self.assertEqual(reader.pcs_count, 1)
        np.testing.assert_array_almost_equal(reader.get_pc_xyz(0), xyz)
        self.assertEqual(reader.get_pc_reference_frame_id(0), "ego")
        self.assertEqual(reader.get_pc_reference_frame_timestamp_us(0), 500_000)
        self.assertEqual(reader.attribute_names, [])
        self.assertEqual(reader.get_pc_generic_data_names(0), [])
        self.assertEqual(reader.get_pc_generic_meta_data(0), {})

        tmpdir.cleanup()

    def test_generic_data_and_metadata(self):
        """Verify generic_data arrays and generic_meta_data round-trip."""
        pc_writer, store_writer, tmpdir, _ = self._make_writer_reader()

        xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        gd_labels = np.array([42], dtype=np.int32)
        gd_weights = np.array([0.95], dtype=np.float64)
        gmd: Dict[str, JsonLike] = {"source": "lidar_top", "quality": 0.99, "tags": ["outdoor", "sunny"]}

        pc_writer.store_pc(
            xyz=xyz,
            reference_frame_id="world",
            reference_frame_timestamp_us=1_000_000,
            generic_data={"labels": gd_labels, "weights": gd_weights},
            generic_meta_data=gmd,
        )

        reader = self._finalize_and_open_reader(store_writer)

        # generic_data
        self.assertEqual(sorted(reader.get_pc_generic_data_names(0)), ["labels", "weights"])
        self.assertTrue(reader.has_pc_generic_data(0, "labels"))
        self.assertFalse(reader.has_pc_generic_data(0, "nonexistent"))
        np.testing.assert_array_equal(reader.get_pc_generic_data(0, "labels"), gd_labels)
        np.testing.assert_array_almost_equal(reader.get_pc_generic_data(0, "weights"), gd_weights)

        # generic_meta_data
        loaded_gmd = reader.get_pc_generic_meta_data(0)
        self.assertEqual(loaded_gmd["source"], "lidar_top")
        self.assertAlmostEqual(cast(float, loaded_gmd["quality"]), 0.99)
        self.assertEqual(loaded_gmd["tags"], ["outdoor", "sunny"])

        tmpdir.cleanup()
