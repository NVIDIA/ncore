# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import unittest
import random
import itertools
import tempfile
import io

from pathlib import Path

import numpy as np
import PIL.Image as PILImage
import parameterized

from scipy.spatial.transform import Rotation as R

from .data3 import ShardDataLoader, Sensor, CameraSensor, LidarSensor, ContainerDataWriter
from .types import FrameTimepoint, FThetaCameraModelParameters, OpenCVFisheyeCameraModelParameters, Poses, ShutterType


class TestData3Loader(unittest.TestCase):
    """Test to verify functionality of V3 data loader"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        self.random = random.Random(x=0)  # seed deterministically
        self.all_shards = sorted([str(p) for p in Path("external/test-data-v3-shards").iterdir() if p.match("*.itar")])

    @parameterized.parameterized.expand(itertools.product((False, True), (False, True)))
    def test_shard_loader(self, open_consolidated: bool, reload_store_resources: bool):
        shard_num_poses = [5, 4, 3]
        self.assertEqual(len(self.all_shards), 3)

        def check(start, stop):
            # Randomize shard path order
            local_shards = self.all_shards[start:stop]
            self.random.shuffle(local_shards)

            loader = ShardDataLoader(local_shards, open_consolidated=open_consolidated)

            if reload_store_resources:
                loader.reload_store_resources()

            # expected number of total poses is sum of per-shard poses minus duplicated/removed poses at shard boundaries
            expected_num_poses = sum(shard_num_poses[start:stop]) - (stop - start - 1)

            self.assertEqual(len(loader.get_camera_ids()), 12)
            self.assertEqual(len(loader.get_lidar_ids()), 1)
            self.assertEqual(len(loader.get_radar_ids()), 18)
            self.assertEqual(len(loader.get_sensor_ids()), 31)

            poses = loader.get_poses()
            self.assertEqual(poses.T_rig_world_base.shape, (4, 4))
            self.assertEqual(poses.T_rig_world_timestamps_us.shape, (expected_num_poses,))
            self.assertEqual(poses.T_rig_worlds.shape, (expected_num_poses, 4, 4))

            for local_shard_idx, shard_id in enumerate(range(start, stop)):
                # check *single* shard sub-range pose lookup
                self.assertEqual(
                    loader.get_poses(local_shard_idx, local_shard_idx + 1).T_rig_world_timestamps_us.shape,
                    (shard_num_poses[shard_id],),
                )

            self.assertEqual(
                loader.get_sequence_id(with_shard_range=False),
                "c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035",
            )
            self.assertEqual(
                loader.get_sequence_id(with_shard_range=True),
                "c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_"
                + "_".join([str(shard_id) for shard_id in range(start, stop)]),
            )

            self.assertTrue("nv-rig" in loader.get_generic_meta_data())

            self.assertEqual(loader.get_calibration_type(), "scene-calib")
            self.assertEqual(loader.get_egomotion_type(), "icp")

            # make sure returned paths are absolute and ordered by shard-id
            self.assertEqual(loader.get_shard_paths(), [str(Path(p).absolute()) for p in self.all_shards[start:stop]])
            self.assertEqual(loader.get_shard_ids(), list(range(start, stop)))

            # check global tracks API
            tracks = loader.get_tracks()
            self.assertEqual(len(tracks.track_labels), 40)

        # check all shard slice variants
        for stop in range(1, len(self.all_shards) + 1):
            for start in range(0, stop):
                check(start, stop)

    def test_shard_sensor_lidar(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(lidar_sensor := loader.get_sensor("lidar_gt_top_p128_v4p5"), LidarSensor)
        self.assertEqual(lidar_sensor.get_sensor_id(), "lidar_gt_top_p128_v4p5")
        self.assertEqual(
            lidar_sensor.get_sensor_id(), loader.get_lidar_sensor("lidar_gt_top_p128_v4p5").get_sensor_id()
        )

        # We currently don't store generic meta data in NV data
        self.assertEqual(lidar_sensor.get_generic_meta_data(), {})

        # Load all data
        for frame_index in lidar_sensor.get_frame_index_range():
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, "xyz_s"))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, "xyz_e"))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, "intensity"))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, "timestamp_us"))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, "dynamic_flag"))
            self.assertFalse(lidar_sensor.has_frame_data(frame_index, "foo"))

            point_count = lidar_sensor.get_frame_data(frame_index, "xyz_e").shape[0]
            self.assertEqual(
                lidar_sensor.get_frame_data(frame_index, "xyz_s").shape,
                lidar_sensor.get_frame_data(frame_index, "xyz_e").shape,
            )
            self.assertEqual(len(lidar_sensor.get_frame_data(frame_index, "intensity")), point_count)
            self.assertEqual(len(lidar_sensor.get_frame_data(frame_index, "timestamp_us")), point_count)
            self.assertEqual(len(lidar_sensor.get_frame_data(frame_index, "dynamic_flag")), point_count)

            self.assertGreater(
                len(lidar_sensor.get_frame_labels(frame_index)), 0, f"no labels for lidar frame {frame_index}"
            )

            # We currently don't store generic frame data in NV data
            self.assertEqual(lidar_sensor.get_frame_generic_meta_data(frame_index), {})
            self.assertEqual(lidar_sensor.get_frame_generic_data_names(frame_index), [])
            self.assertFalse(lidar_sensor.has_frame_generic_data(frame_index, "nonexisting"))
            self.assertRaises(ValueError, lidar_sensor.get_frame_generic_data, frame_index, "nonexisting")

    def test_shard_sensor_camera(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(camera_sensor := loader.get_sensor("camera_cross_left_120fov"), CameraSensor)
        self.assertEqual(camera_sensor.get_sensor_id(), "camera_cross_left_120fov")
        self.assertEqual(
            camera_sensor.get_sensor_id(), loader.get_camera_sensor("camera_cross_left_120fov").get_sensor_id()
        )

        # We currently don't store generic meta data in NV data
        self.assertEqual(camera_sensor.get_generic_meta_data(), {})

        self.assertIsInstance(camera_sensor.get_camera_model_parameters(), FThetaCameraModelParameters)
        self.assertEqual(camera_sensor.get_camera_mask_image().size, (3848, 2168))

        # Decode all camera frames
        for frame_index in camera_sensor.get_frame_index_range():
            self.assertEqual(camera_sensor.get_frame_image(frame_index).size, (3848, 2168))

            # We currently don't store generic frame data in NV data
            self.assertEqual(camera_sensor.get_frame_generic_meta_data(frame_index), {})
            self.assertEqual(camera_sensor.get_frame_generic_data_names(frame_index), [])
            self.assertFalse(camera_sensor.has_frame_generic_data(frame_index, "nonexisting"))
            self.assertRaises(ValueError, camera_sensor.get_frame_generic_data, frame_index, "nonexisting")

    def test_shard_sensor(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(sensor := loader.get_sensor("camera_front_wide_120fov"), Sensor)
        self.assertEqual(sensor.get_sensor_id(), "camera_front_wide_120fov")

        # Check some known values across shards
        reference_T_sensor_rig = np.array(
            [
                [-0.01506471, -0.0072778263, 0.99986, 1.774368],
                [-0.9998305, 0.010698613, -0.014986393, 0.0035241419],
                [-0.010588046, -0.9999163, -0.0074377647, 1.4483173],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertIsNone(np.testing.assert_array_equal(sensor.get_T_sensor_rig(), reference_T_sensor_rig))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(sensor.get_T_rig_sensor(), np.linalg.inv(reference_T_sensor_rig))
        )

        self.assertEqual(sensor.get_frames_count(), 27)

        self.assertEqual(sensor.get_frames_count(0, 1), 12)
        self.assertEqual(sensor.get_frames_count(1, 2), 9)
        self.assertEqual(sensor.get_frames_count(2, 3), 6)
        self.assertEqual(sensor.get_frames_count(0, 2), 21)

        self.assertEqual(len(sensor.get_frames_timestamps_us(0, 1)), 12)
        self.assertEqual(len(sensor.get_frames_timestamps_us(1, 2)), 9)
        self.assertEqual(len(sensor.get_frames_timestamps_us(2, 3)), 6)
        self.assertEqual(len(sensor.get_frames_timestamps_us(0, 2)), 21)

        self.assertEqual(sensor.get_frame_index_range(), range(0, 27, 1))
        self.assertEqual(sensor.get_frame_index_range(None, None, None), range(0, 27, 1))
        self.assertEqual(sensor.get_frame_index_range(0, 27, 1), range(0, 27, 1))
        self.assertEqual(sensor.get_frame_index_range(0), range(0, 27, 1))
        self.assertEqual(sensor.get_frame_index_range(1, 3, None), range(1, 3, 1))
        self.assertEqual(sensor.get_frame_index_range(1, None, 2), range(1, 27, 2))
        self.assertEqual(sensor.get_frame_index_range(1, -1, 1), range(1, 26, 1))  # negative slice arguments
        self.assertEqual(sensor.get_frame_index_range(-2), range(25, 27, 1))
        self.assertEqual(sensor.get_frame_index_range(-2, None, -1), range(25, -1, -1))
        self.assertEqual(sensor.get_frame_index_range(28), range(27, 27, 1))  # empty ranges
        self.assertEqual(sensor.get_frame_index_range(2 * 28), range(27, 27, 1))

        # Check that all sensor timestamps are strictly monotonically increasing
        self.assertTrue(np.all((timestamps := sensor.get_frames_timestamps_us())[:-1] < timestamps[1:]))

        # first frame in first shard
        reference_T_rig_world_0_start = np.array(
            [
                [0.9966336, 0.0819398, -0.0027226843, -0.06672776],
                [-0.08177382, 0.9959, 0.038681187, -5.527364],
                [0.00588105, -0.038328324, 0.9992479, 0.34190598],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(sensor.get_frames_timestamps_us()[0], 1648597318807685)
        self.assertEqual(sensor.get_closest_frame_index(1648597318807685), 0)
        self.assertEqual(sensor.get_frame_timestamp_us(0, FrameTimepoint.START), 1648597318776074)
        self.assertEqual(sensor.get_frame_timestamp_us(0, FrameTimepoint.END), 1648597318807685)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(0, FrameTimepoint.START), reference_T_rig_world_0_start
            )
        )
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                sensor.get_frame_T_world_rig(0, FrameTimepoint.START), np.linalg.inv(reference_T_rig_world_0_start)
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_sensor_world(0, FrameTimepoint.START),
                reference_T_rig_world_0_start @ reference_T_sensor_rig,
            )
        )
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                sensor.get_frame_T_world_sensor(0, FrameTimepoint.START),
                np.linalg.inv(reference_T_rig_world_0_start @ reference_T_sensor_rig),
            )
        )
        self.assertEqual(sensor.get_frame_image(0).size, (3848, 2168))

        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(0, FrameTimepoint.END),
                np.array(
                    [
                        [0.9942354, 0.10717609, -0.003044726, -0.07839771],
                        [-0.10689344, 0.9930261, 0.04972909, -7.2354264],
                        [0.008353262, -0.04911696, 0.9987581, 0.44561356],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )

        # second frame in first shard
        self.assertEqual(sensor.get_frames_timestamps_us()[1], 1648597318840981)
        self.assertEqual(sensor.get_closest_frame_index(1648597318840981), 1)
        self.assertEqual(sensor.get_frame_timestamp_us(1, FrameTimepoint.START), 1648597318809370)
        self.assertEqual(sensor.get_frame_timestamp_us(1, FrameTimepoint.END), 1648597318840981)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(1, FrameTimepoint.START),
                np.array(
                    [
                        [0.99424964, 0.10704328, -0.0030718392, -0.076306164],
                        [-0.106761694, 0.9930536, 0.049461745, -7.226634],
                        [0.008345049, -0.048849367, 0.9987713, 0.44463754],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(1, FrameTimepoint.END),
                np.array(
                    [
                        [0.9945132, 0.10455002, -0.0035738598, -0.03706818],
                        [-0.104289405, 0.9935534, 0.044444963, -7.0616794],
                        [0.008197542, -0.043828387, 0.99900544, 0.42632708],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )

        # first frame in second shard
        self.assertEqual(sensor.get_frames_timestamps_us()[4], 1648597318940968)
        self.assertEqual(sensor.get_closest_frame_index(1648597318940968), 4)
        self.assertEqual(sensor.get_frame_timestamp_us(4, FrameTimepoint.START), 1648597318909357)
        self.assertEqual(sensor.get_frame_timestamp_us(4, FrameTimepoint.END), 1648597318940968)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(4, FrameTimepoint.START),
                np.array(
                    [
                        [0.99483097, 0.101455204, -0.004267634, 0.017603396],
                        [-0.10123113, 0.9941819, 0.03680526, -6.8479548],
                        [0.00797689, -0.036182996, 0.99931335, 0.39923683],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(4, FrameTimepoint.END),
                np.array(
                    [
                        [0.9942722, 0.1068194, -0.003527579, -0.046103783],
                        [-0.106572434, 0.9933853, 0.042753946, -7.170697],
                        [0.008071195, -0.042133115, 0.9990794, 0.42358804],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )

        # second frame in second shard
        self.assertEqual(sensor.get_frames_timestamps_us()[5], 1648597318974378)
        self.assertEqual(sensor.get_closest_frame_index(1648597318974378), 5)
        self.assertEqual(sensor.get_frame_timestamp_us(5, FrameTimepoint.START), 1648597318942767)
        self.assertEqual(sensor.get_frame_timestamp_us(5, FrameTimepoint.END), 1648597318974378)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(5, FrameTimepoint.START),
                np.array(
                    [
                        [0.9942395, 0.10712445, -0.0034844966, -0.049729396],
                        [-0.1068763, 0.99333805, 0.04309232, -7.1890645],
                        [0.008077525, -0.04247168, 0.99906504, 0.42497388],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(5, FrameTimepoint.END),
                np.array(
                    [
                        [0.9936502, 0.11248058, -0.0027105322, -0.11343657],
                        [-0.11221362, 0.9924735, 0.049035087, -7.511807],
                        [0.008205626, -0.048419565, 0.99879336, 0.44932508],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )

        # first frame in third shard
        self.assertEqual(sensor.get_frames_timestamps_us()[8], 1648597319074372)
        self.assertEqual(sensor.get_closest_frame_index(1648597319074372), 8)
        self.assertEqual(sensor.get_frame_timestamp_us(8, FrameTimepoint.START), 1648597319042761)
        self.assertEqual(sensor.get_frame_timestamp_us(8, FrameTimepoint.END), 1648597319074372)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(8, FrameTimepoint.START),
                np.array(
                    [
                        [0.9942919, 0.106653325, -0.0029403504, -0.06390721],
                        [-0.10646834, 0.99360436, 0.037614945, -7.0825877],
                        [0.006933304, -0.037087183, 0.99928796, 0.38593167],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(8, FrameTimepoint.END),
                np.array(
                    [
                        [0.9950704, 0.09910973, -0.0034858456, 0.011042176],
                        [-0.09898916, 0.9947596, 0.025582034, -6.5709286],
                        [0.006003007, -0.025110865, 0.99966663, 0.32437676],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


class TestData3Reload(unittest.TestCase):
    """Test to verify functionality of V3 data writer + loader"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

    def test_reload(self):
        """Test to make sure serialized data is faithfully reloaded"""
        tempdir = tempfile.TemporaryDirectory()

        ## Create reference data / shard

        writer = ContainerDataWriter(
            Path(tempdir.name),
            ref_sequence_id := "some-sequence-name",
            [ref_camera_id := "camera-sensor"],
            [ref_lidar_id := "lidar-sensor"],
            [ref_radar_id := "radar-sensor"],
            # TODO: parse these from the data
            ref_calibration_type := "some-calibration-type",
            ref_egomotion_type := "some-egomotion-type",
            ref_sequence_id,
            ref_generic_meta_data := {"some-meta": "data"},
            # always single-shard
            0,
            1,
            False,
        )

        # Store poses
        base_pose = np.eye(4, dtype=np.float64)

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
        T_rig_world_timestamps_us = np.array([0 * 1e6, 0.1 * 1e6], dtype=np.uint64)

        ref_poses = Poses(
            T_rig_world_base=base_pose,
            T_rig_worlds=T_rig_worlds.astype(np.float64),
            T_rig_world_timestamps_us=T_rig_world_timestamps_us,
        )

        writer.store_poses(ref_poses)

        # Store camera data
        writer.store_camera_meta(
            ref_camera_id,
            T_rig_world_timestamps_us,
            ref_camera_extrinsics := np.block(
                [
                    [R.from_euler("xyz", [1, 1.1, 2.2], degrees=True).as_matrix(), np.array([1, 0, 0]).reshape((3, 1))],
                    [np.array([0, 0, 0, 1])],
                ]
            ).astype(np.float32),
            OpenCVFisheyeCameraModelParameters(
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
            ),
            None,
            ref_camera_generic_meta_data := {"some-meta-data": np.random.rand(3, 2).tolist()},
        )

        with io.BytesIO() as buffer:
            PILImage.fromarray(ref_image_rgb0 := np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)).save(
                buffer, format="png", optimize=True, quality=91
            )

            writer.store_camera_frame(
                ref_camera_id,
                0,
                buffer.getvalue(),
                "png",
                T_rig_worlds.astype(np.float32),
                T_rig_world_timestamps_us,
                ref_camera_generic_data0 := {"some-frame-data": np.random.rand(6, 2)},
                ref_camera_generic_metadata0 := {"some-frame-meta-data": {"something": 1, "else": 2}},
            )

        with io.BytesIO() as buffer:
            PILImage.fromarray(ref_image_rgb1 := np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)).save(
                buffer, format="png", optimize=True, quality=91
            )

            writer.store_camera_frame(
                ref_camera_id,
                1,
                buffer.getvalue(),
                "png",
                T_rig_worlds.astype(np.float32),
                T_rig_world_timestamps_us,
                ref_camera_generic_data1 := {"some-frame-data": np.random.rand(6, 2)},
                ref_camera_generic_metadata1 := {"some-more-frame-meta-data": {"even": True, "more": None}},
            )

        # Store lidar data
        writer.store_lidar_meta(
            ref_lidar_id,
            T_rig_world_timestamps_us,
            ref_lidar_extrinsics := np.block(
                [
                    [R.from_euler("xyz", [5, 1.1, 2.2], degrees=True).as_matrix(), np.array([1, 5, 0]).reshape((3, 1))],
                    [np.array([0, 0, 0, 1])],
                ]
            ).astype(np.float32),
            ref_lidar_generic_meta_data := {"some-meta-data": np.random.rand(3, 2).tolist()},
        )

        writer.store_lidar_frame(
            ref_lidar_id,
            0,
            ref_lidar_xyz_s0 := np.random.rand(5, 3).astype(np.float32),
            ref_lidar_xyz_e0 := np.random.rand(5, 3).astype(np.float32),
            ref_lidar_intensity0 := np.random.rand(5).astype(np.float32),
            ref_lidar_timestamp_us0 := np.array([1, 2, 3, 4, 5], dtype=np.uint64),
            [],
            T_rig_worlds.astype(np.float32),
            T_rig_world_timestamps_us,
            ref_lidar_generic_data0 := {
                "some-frame-data": np.random.rand(6, 3),
                "dynamic_flag": np.random.rand(5).astype(np.int8),
            },
            ref_lidar_generic_metadata0 := {"some-frame-meta-data": {"random-data": np.random.rand(6, 3).tolist()}},
        )

        writer.store_lidar_frame(
            ref_lidar_id,
            1,
            ref_lidar_xyz_s1 := np.random.rand(5, 3).astype(np.float32),
            ref_lidar_xyz_e1 := np.random.rand(5, 3).astype(np.float32),
            ref_lidar_intensity1 := np.random.rand(5).astype(np.float32),
            ref_lidar_timestamp_us1 := np.array([1, 2, 3, 4, 5], dtype=np.uint64),
            [],
            T_rig_worlds.astype(np.float32),
            T_rig_world_timestamps_us,
            ref_lidar_generic_data1 := {
                "some-frame-data": np.random.rand(6, 3),
                "dynamic_flag": np.random.rand(5).astype(np.int8),
            },
            ref_lidar_generic_metadata1 := {"some-frame-meta-data": {"random-data": np.random.rand(6, 3).tolist()}},
        )

        # Store radar data
        writer.store_radar_meta(
            ref_radar_id,
            T_rig_world_timestamps_us,
            ref_radar_extrinsics := np.block(
                [
                    [R.from_euler("xyz", [5, 1.1, 2.2], degrees=True).as_matrix(), np.array([1, 5, 0]).reshape((3, 1))],
                    [np.array([0, 0, 0, 1])],
                ]
            ).astype(np.float32),
            ref_radar_generic_meta_data := {"some-meta-data": np.random.rand(3, 2).tolist()},
        )

        writer.store_radar_frame(
            ref_radar_id,
            0,
            ref_radar_xyz_s0 := np.random.rand(5, 3).astype(np.float32),
            ref_radar_xyz_e0 := np.random.rand(5, 3).astype(np.float32),
            T_rig_worlds.astype(np.float32),
            T_rig_world_timestamps_us,
            ref_radar_generic_data0 := {"some-frame-data": np.random.rand(6, 3)},
            ref_radar_generic_metadata0 := {"some-frame-meta-data": {"random-data": np.random.rand(6, 3).tolist()}},
        )

        writer.store_radar_frame(
            ref_radar_id,
            1,
            ref_radar_xyz_s1 := np.random.rand(5, 3).astype(np.float32),
            ref_radar_xyz_e1 := np.random.rand(5, 3).astype(np.float32),
            T_rig_worlds.astype(np.float32),
            T_rig_world_timestamps_us,
            ref_radar_generic_data1 := {"some-frame-data": np.random.rand(6, 3)},
            ref_radar_generic_metadata1 := {"some-frame-meta-data": {"random-data": np.random.rand(6, 3).tolist()}},
        )

        shard_path = writer.finalize()

        ## Reload shard and verify consistency
        loader = ShardDataLoader([str(shard_path)])

        self.assertEqual(loader.get_sequence_id(), ref_sequence_id)
        self.assertEqual(loader.get_generic_meta_data(), ref_generic_meta_data)
        self.assertEqual(loader.get_calibration_type(), ref_calibration_type)
        self.assertEqual(loader.get_egomotion_type(), ref_egomotion_type)

        # Check poses
        self.assertIsNone(
            np.testing.assert_array_equal((poses := loader.get_poses()).T_rig_world_base, ref_poses.T_rig_world_base)
        )
        self.assertIsNone(
            np.testing.assert_array_equal(poses.T_rig_world_timestamps_us, ref_poses.T_rig_world_timestamps_us)
        )
        self.assertIsNone(np.testing.assert_array_equal(poses.T_rig_worlds, ref_poses.T_rig_worlds))

        # Check cameras
        camera_sensor = loader.get_camera_sensor(ref_camera_id)
        self.assertIsNone(np.testing.assert_array_equal(camera_sensor.get_T_sensor_rig(), ref_camera_extrinsics))
        self.assertEqual(camera_sensor.get_generic_meta_data(), ref_camera_generic_meta_data)

        self.assertIsNone(np.testing.assert_array_equal(camera_sensor.get_frame_image_array(0), ref_image_rgb0))
        self.assertEqual(camera_sensor.get_frame_generic_meta_data(0), ref_camera_generic_metadata0)
        self.assertEqual(names := camera_sensor.get_frame_generic_data_names(0), list(ref_camera_generic_data0.keys()))
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    camera_sensor.get_frame_generic_data(0, name), ref_camera_generic_data0[name]
                )
            )

        self.assertIsNone(np.testing.assert_array_equal(camera_sensor.get_frame_image_array(1), ref_image_rgb1))
        self.assertEqual(camera_sensor.get_frame_generic_meta_data(1), ref_camera_generic_metadata1)
        self.assertEqual(names := camera_sensor.get_frame_generic_data_names(1), list(ref_camera_generic_data1.keys()))
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    camera_sensor.get_frame_generic_data(1, name), ref_camera_generic_data1[name]
                )
            )

        # Check lidars
        lidar_sensor = loader.get_lidar_sensor(ref_lidar_id)
        self.assertIsNone(np.testing.assert_array_equal(lidar_sensor.get_T_sensor_rig(), ref_lidar_extrinsics))
        self.assertEqual(lidar_sensor.get_generic_meta_data(), ref_lidar_generic_meta_data)

        self.assertIsNone(np.testing.assert_array_equal(lidar_sensor.get_frame_data(0, "xyz_s"), ref_lidar_xyz_s0))
        self.assertIsNone(np.testing.assert_array_equal(lidar_sensor.get_frame_data(0, "xyz_e"), ref_lidar_xyz_e0))
        self.assertIsNone(
            np.testing.assert_array_equal(lidar_sensor.get_frame_data(0, "intensity"), ref_lidar_intensity0)
        )
        self.assertIsNone(
            np.testing.assert_array_equal(lidar_sensor.get_frame_data(0, "timestamp_us"), ref_lidar_timestamp_us0)
        )
        self.assertFalse(lidar_sensor.has_frame_data(0, "dynamic_flag"))  # deprecated property
        self.assertEqual(lidar_sensor.get_frame_generic_meta_data(0), ref_lidar_generic_metadata0)
        self.assertEqual(
            names := sorted(lidar_sensor.get_frame_generic_data_names(0)), sorted(list(ref_lidar_generic_data0.keys()))
        )
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    lidar_sensor.get_frame_generic_data(0, name), ref_lidar_generic_data0[name]
                )
            )

        self.assertIsNone(np.testing.assert_array_equal(lidar_sensor.get_frame_data(1, "xyz_s"), ref_lidar_xyz_s1))
        self.assertIsNone(np.testing.assert_array_equal(lidar_sensor.get_frame_data(1, "xyz_e"), ref_lidar_xyz_e1))
        self.assertIsNone(
            np.testing.assert_array_equal(lidar_sensor.get_frame_data(1, "intensity"), ref_lidar_intensity1)
        )
        self.assertIsNone(
            np.testing.assert_array_equal(lidar_sensor.get_frame_data(1, "timestamp_us"), ref_lidar_timestamp_us1)
        )
        self.assertFalse(lidar_sensor.has_frame_data(1, "dynamic_flag"))  # deprecated property
        self.assertEqual(lidar_sensor.get_frame_generic_meta_data(1), ref_lidar_generic_metadata1)
        self.assertEqual(
            names := sorted(lidar_sensor.get_frame_generic_data_names(1)), sorted(list(ref_lidar_generic_data1.keys()))
        )
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    lidar_sensor.get_frame_generic_data(1, name), ref_lidar_generic_data1[name]
                )
            )

        # Check radars
        radar_sensor = loader.get_radar_sensor(ref_radar_id)
        self.assertIsNone(np.testing.assert_array_equal(radar_sensor.get_T_sensor_rig(), ref_radar_extrinsics))
        self.assertEqual(radar_sensor.get_generic_meta_data(), ref_radar_generic_meta_data)

        self.assertIsNone(np.testing.assert_array_equal(radar_sensor.get_frame_data(0, "xyz_s"), ref_radar_xyz_s0))
        self.assertIsNone(np.testing.assert_array_equal(radar_sensor.get_frame_data(0, "xyz_e"), ref_radar_xyz_e0))
        self.assertEqual(radar_sensor.get_frame_generic_meta_data(0), ref_radar_generic_metadata0)
        self.assertEqual(names := radar_sensor.get_frame_generic_data_names(0), list(ref_radar_generic_data0.keys()))
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    radar_sensor.get_frame_generic_data(0, name), ref_radar_generic_data0[name]
                )
            )

        self.assertIsNone(np.testing.assert_array_equal(radar_sensor.get_frame_data(1, "xyz_s"), ref_radar_xyz_s1))
        self.assertIsNone(np.testing.assert_array_equal(radar_sensor.get_frame_data(1, "xyz_e"), ref_radar_xyz_e1))
        self.assertEqual(radar_sensor.get_frame_generic_meta_data(1), ref_radar_generic_metadata1)
        self.assertEqual(names := radar_sensor.get_frame_generic_data_names(1), list(ref_radar_generic_data1.keys()))
        for name in names:
            self.assertIsNone(
                np.testing.assert_array_equal(
                    radar_sensor.get_frame_generic_data(1, name), ref_radar_generic_data1[name]
                )
            )
