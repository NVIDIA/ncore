# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import io
import unittest
import tempfile

from pathlib import Path

import numpy as np
import PIL.Image as PILImage

from parameterized import parameterized
from scipy.spatial.transform import Rotation as R

from .components import (
    LidarSensorComponent,
    SequenceComponentStoreWriter,
    SequenceComponentStoreReader,
    PosesSetComponent,
    CameraSensorComponent,
    SensorIntrinsicsComponent,
    CuboidTracksComponent,
)
from .types import CuboidTrack
from ncore.impl.data.types import (
    OpenCVFisheyeCameraModelParameters,
    ShutterType,
    BivariateWindshieldModelParameters,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    LabelSource,
    BBox3,
)


class TestData4Reload(unittest.TestCase):
    """Test to verify functionality of V4 data writer + loader"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

    @parameterized.expand(
        [
            ("itar", False),
            ("itar", True),
            ("directory", False),
            ("directory", True),
        ]
    )
    def test_reload(self, store_type, open_consolidated):
        """Test to make sure serialized data is faithfully reloaded"""

        # from scripts.util import breakpoint
        # breakpoint()

        tempdir = tempfile.TemporaryDirectory()

        ## Create reference sequence
        store_writer = SequenceComponentStoreWriter(
            output_dir_path=Path(tempdir.name),
            store_base_name=(ref_sequence_id := "some-sequence-name"),
            sequence_id=ref_sequence_id,
            store_type=store_type,
            generic_meta_data=(ref_generic_sequence_meta_data := {"some": 1, "key": 1.2}),
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
            ]
        )
        T_rig_world_timestamps_us = np.array([0 * 1e6, 0.2 * 1e6], dtype=np.uint64)

        store_writer.register_component_writer(
            PosesSetComponent.Writer,
            ref_poses_id := "some_poses_type",
            group_name=None,  # use default component group
            generic_meta_data=(ref_pose_generic_meta_data := {"some": "thing"}),
        ).store_dynamic_poses(
            source_frame="rig",
            target_frame="world",
            poses=(ref_T_rig_worlds := T_rig_worlds),
            timestamps_us=(ref_T_rig_world_timestamps_us := T_rig_world_timestamps_us),
        ).store_static_pose(
            source_frame="world",
            target_frame="world_global",
            pose=(ref_T_world_world_global := T_world_world_global),
        ).store_static_pose(
            source_frame=(ref_camera_id := "ref_camera_id"),
            target_frame="rig",
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
        )

        # Store intrinsics
        intrinsic_writer = store_writer.register_component_writer(
            SensorIntrinsicsComponent.Writer, ref_intrinsics_id := "default", "intrinsics"
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
            ref_camera_mask_image := PILImage.fromarray(np.random.rand(3840, 2160) > 0.5).resize((480, 270)),
        )

        intrinsic_writer.store_lidar_intrinsics(
            ref_lidar_id := "some-lidar-sensor-name",
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

        # Store camera data
        camera_writer = store_writer.register_component_writer(
            CameraSensorComponent.Writer,
            ref_camera_id,
            "cameras",
            ref_camera_generic_meta_data := {"some-meta-data": np.random.rand(3, 2).tolist()},
        )

        with io.BytesIO() as buffer:
            PILImage.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)).save(
                buffer, format="png", optimize=True, quality=91
            )

            camera_writer.store_frame(
                ref_image_binary0 := buffer.getvalue(),
                "png",
                ref_camera_timestamps_us0 := np.array([0 * 1e6, 0.1 * 1e6], dtype=np.uint64),
                ref_camera_generic_data0 := {"some-frame-data": np.random.rand(6, 2)},
                ref_camera_generic_metadata0 := {"some-frame-meta-data": {"something": 1, "else": 2}},
            )

        with io.BytesIO() as buffer:
            PILImage.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)).save(
                buffer, format="png", optimize=True, quality=91
            )

            camera_writer.store_frame(
                ref_image_binary1 := buffer.getvalue(),
                "png",
                ref_camera_timestamps_us1 := np.array([0.1 * 1e6, 0.2 * 1e6], dtype=np.uint64),
                ref_camera_generic_data1 := {"some-frame-data": np.random.rand(6, 2)},
                ref_camera_generic_metadata1 := {"some-more-frame-meta-data": {"even": True, "more": None}},
            )

        # Store lidar data
        lidar_writer = store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            ref_lidar_id,
            "lidars",
            ref_lidar_generic_meta_data := {"some-lidar-meta-data": np.random.rand(3, 2).tolist()},
        )

        lidar_writer.store_frame(
            ref_lidar_xyz_m0 := np.random.rand(5, 3).astype(np.float32) + 0.1,
            ref_lidar_intensity0 := np.random.rand(5).astype(np.float32),
            ref_lidar_timestamp_us0 := np.linspace(0 * 1e6, 0.5 * 1e6, num=5, dtype=np.uint64),
            ref_lidar_model_element0 := np.arange(5 * 2, dtype=np.uint16).reshape((5, 2)),
            ref_lidar_timestamps_us0 := np.array([0 * 1e6, 0.5 * 1e6], dtype=np.uint64),
            ref_lidar_generic_data0 := {"some-other-frame-data": np.random.rand(6, 2)},
            ref_lidar_generic_metadata0 := {"some-more-meta-data": {"yes": None, "no": True}},
        )

        lidar_writer.store_frame(
            ref_lidar_xyz_m1 := np.random.rand(8, 3).astype(np.float32) + 0.1,
            ref_lidar_intensity1 := np.random.rand(8).astype(np.float32),
            ref_lidar_timestamp_us1 := np.linspace(0.5 * 1e6, 1 * 1e6, num=8, dtype=np.uint64),
            ref_lidar_model_element1 := np.arange(8 * 2, dtype=np.uint16).reshape((8, 2)),
            ref_lidar_timestamps_us1 := np.array([0.5 * 1e6, 1 * 1e6], dtype=np.uint64),
            ref_lidar_generic_data1 := {"some-other-frame-data": np.random.rand(2, 2)},
            ref_lidar_generic_metadata1 := {"even-more-meta-data": {"yesno": None}},
        )

        # Store cuboid tracks
        tracks_writer = store_writer.register_component_writer(
            CuboidTracksComponent.Writer,
            ref_tracks_id := "default",
            "tracks",
            ref_tracks_generic_meta_data := {"track-set-meta-data": "some-value"},
        )

        tracks_writer.store_tracks(
            cuboid_tracks=(
                ref_cuboid_tracks := [
                    CuboidTrack(
                        track_id="track-1",
                        label_class="car",
                        reference_frame_name=ref_lidar_id,
                        source=LabelSource.AUTOLABEL,
                        source_version="v0",
                        observations=[
                            CuboidTrack.Observation(
                                observation_id="obs-1-1",
                                timestamp_us=int(0.3 * 1e6),
                                reference_frame_timestamp_us=int(0.5 * 1e6),
                                bbox3=BBox3(
                                    centroid=(1.0, 2.0, 3.0),
                                    dim=(4.0, 2.0, 1.5),
                                    rot=(0.0, 0.0, 0.0),
                                ),
                            ),
                            CuboidTrack.Observation(
                                observation_id="obs-1-2",
                                timestamp_us=int(0.4 * 1e6),
                                reference_frame_timestamp_us=int(1.0 * 1e6),
                                bbox3=BBox3(
                                    centroid=(1.5, 2.5, 3.5),
                                    dim=(4.0, 2.0, 1.5),
                                    rot=(0.0, 0.0, 0.1),
                                ),
                            ),
                        ],
                    )
                ]
            )
        )

        ## Finalize writers
        store_paths = store_writer.finalize()

        ## Reload sequence and verify consistency
        store_reader = SequenceComponentStoreReader(store_paths, open_consolidated=open_consolidated)

        # check sequence data
        self.assertEqual(store_reader.sequence_id, ref_sequence_id)
        self.assertEqual(store_reader.generic_meta_data, ref_generic_sequence_meta_data)

        # check rig pose / calibration data
        poses_readers = store_reader.open_component_readers(PosesSetComponent.Reader)

        self.assertEqual(len(poses_readers), 1)
        poses_reader = poses_readers[ref_poses_id]

        self.assertEqual(poses_reader.instance_name, ref_poses_id)
        self.assertEqual(poses_reader.generic_meta_data, ref_pose_generic_meta_data)

        self.assertTrue(np.all(poses_reader.get_static_pose("world", "world_global") == ref_T_world_world_global))

        T_rig_worlds, T_rig_world_timestamps_us = poses_reader.get_dynamic_poses("rig", "world")
        self.assertTrue(np.all(T_rig_worlds == ref_T_rig_worlds))
        self.assertTrue(np.all(T_rig_world_timestamps_us == ref_T_rig_world_timestamps_us))

        self.assertTrue(np.all(poses_reader.get_static_pose(ref_camera_id, "rig") == ref_camera_T_sensor_rig))

        with self.assertRaises(KeyError):
            poses_reader.get_static_pose("non-existing-sensor", "rig")

        with self.assertRaises(KeyError):
            poses_reader.get_dynamic_poses("non-existing-frame", "world")

        # check intrinsics data
        intrinsic_readers = store_reader.open_component_readers(SensorIntrinsicsComponent.Reader)

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
            intrinsic_reader.get_camera_mask_image(ref_camera_id).tobytes(), ref_camera_mask_image.tobytes()
        )

        self.assertEqual(
            (lidar_model_parameters := intrinsic_reader.get_lidar_model_parameters(ref_lidar_id)).to_dict(),
            ref_lidar_intrinsics.to_dict(),
        )
        self.assertIsInstance(lidar_model_parameters, RowOffsetStructuredSpinningLidarModelParameters)

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
            camera_reader.get_frame_image_binary_data(ref_camera_timestamps_us0[1]), (ref_image_binary0, "png")
        )
        self.assertEqual(
            camera_reader.get_frame_image_binary_data(ref_camera_timestamps_us1[1]), (ref_image_binary1, "png")
        )

        self.assertEqual(
            names := camera_reader.get_frame_generic_data_names(ref_camera_timestamps_us0[1]),
            list(ref_camera_generic_data0.keys()),
        )
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    camera_reader.get_frame_generic_data(ref_camera_timestamps_us0[1], name),
                    ref_camera_generic_data0[name],
                )
            )
        self.assertEqual(
            names := camera_reader.get_frame_generic_data_names(ref_camera_timestamps_us1[1]),
            list(ref_camera_generic_data1.keys()),
        )
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    camera_reader.get_frame_generic_data(ref_camera_timestamps_us1[1], name),
                    ref_camera_generic_data1[name],
                )
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

        self.assertEqual(lidar_reader.get_frame_point_cloud_size(ref_lidar_timestamps_us0[1]), 5)
        self.assertEqual(lidar_reader.get_frame_point_cloud_size(ref_lidar_timestamps_us1[1]), 8)

        ref_point_cloud_data_names = [
            "xyz_m",
            "intensity",
            "timestamp_us",
            "model_element",
        ]
        self.assertEqual(
            set(lidar_reader.get_frame_point_cloud_data_names(ref_lidar_timestamps_us0[1])),
            set(ref_point_cloud_data_names),
        )
        self.assertEqual(
            set(lidar_reader.get_frame_point_cloud_data_names(ref_lidar_timestamps_us1[1])),
            set(ref_point_cloud_data_names),
        )

        for name in ref_point_cloud_data_names:
            self.assertTrue(lidar_reader.has_frame_point_cloud_data(ref_lidar_timestamps_us0[1], name))
            self.assertTrue(lidar_reader.has_frame_point_cloud_data(ref_lidar_timestamps_us1[1], name))

        self.assertTrue(
            np.all(lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us0[1], "xyz_m") == ref_lidar_xyz_m0)
        )
        self.assertTrue(
            np.all(lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us1[1], "xyz_m") == ref_lidar_xyz_m1)
        )

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us0[1], "intensity")
                == ref_lidar_intensity0
            )
        )
        self.assertTrue(
            np.all(
                lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us1[1], "intensity")
                == ref_lidar_intensity1
            )
        )

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us0[1], "timestamp_us")
                == ref_lidar_timestamp_us0
            )
        )
        self.assertTrue(
            np.all(
                lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us1[1], "timestamp_us")
                == ref_lidar_timestamp_us1
            )
        )

        self.assertTrue(
            np.all(
                lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us0[1], "model_element")
                == ref_lidar_model_element0
            )
        )
        self.assertTrue(
            np.all(
                lidar_reader.get_frame_point_cloud_data(ref_lidar_timestamps_us1[1], "model_element")
                == ref_lidar_model_element1
            )
        )

        self.assertEqual(
            names := lidar_reader.get_frame_generic_data_names(ref_lidar_timestamps_us0[1]),
            list(ref_lidar_generic_data0.keys()),
        )
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    lidar_reader.get_frame_generic_data(ref_lidar_timestamps_us0[1], name),
                    ref_lidar_generic_data0[name],
                )
            )
        self.assertEqual(
            names := lidar_reader.get_frame_generic_data_names(ref_lidar_timestamps_us1[1]),
            list(ref_lidar_generic_data1.keys()),
        )
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    lidar_reader.get_frame_generic_data(ref_lidar_timestamps_us1[1], name),
                    ref_lidar_generic_data1[name],
                )
            )

        self.assertEqual(
            lidar_reader.get_frame_generic_meta_data(ref_lidar_timestamps_us0[1]), ref_lidar_generic_metadata0
        )
        self.assertEqual(
            lidar_reader.get_frame_generic_meta_data(ref_lidar_timestamps_us1[1]), ref_lidar_generic_metadata1
        )

        # check tracks data
        tracks_readers = store_reader.open_component_readers(CuboidTracksComponent.Reader)

        self.assertEqual(len(tracks_readers), 1)
        tracks_reader = tracks_readers[ref_tracks_id]
        self.assertEqual(tracks_reader.instance_name, ref_tracks_id)
        self.assertEqual(tracks_reader.generic_meta_data, ref_tracks_generic_meta_data)

        self.assertEqual(tracks_reader.get_tracks(), ref_cuboid_tracks)
