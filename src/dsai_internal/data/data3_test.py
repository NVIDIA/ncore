# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import unittest

from pathlib import Path

import numpy as np

from .data3 import ShardDataLoader, Sensor, CameraSensor, LidarSensor
from .types import FrameTimepoint, FThetaCameraModelParameters


class TestData3Loader(unittest.TestCase):
    ''' Test to verify functionality of V3 data loader '''
    def setUp(self):
        self.all_shards = sorted([p for p in Path('external/test-data-v3-shards').iterdir() if p.match('*.hdf5')])

    def test_shard_loader(self):
        shard_num_poses = [4, 4, 3]
        self.assertEqual(len(self.all_shards), 3)

        def check(start, end):
            loader = ShardDataLoader(self.all_shards[start:end])
            expected_num_poses = sum(shard_num_poses[start:end])

            self.assertEqual(len(loader.get_camera_ids()), 10)
            self.assertEqual(len(loader.get_lidar_ids()), 1)
            self.assertEqual(len(loader.get_radar_ids()), 0)

            poses = loader.get_poses()
            self.assertEqual(poses.T_rig_world_base.shape, (4, 4))
            self.assertEqual(poses.T_rig_world_timestamps_us.shape, (expected_num_poses, ))
            self.assertEqual(poses.T_rig_worlds.shape, (expected_num_poses, 4, 4))

            self.assertEqual(len(np.unique(poses.T_rig_world_timestamps_us)), expected_num_poses)

        # check all shard slice variants
        for end in range(1, len(self.all_shards) + 1):
            for start in range(0, end):
                check(start, end)

    def test_shard_sensor_lidar(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(lidar_sensor := loader.get_sensor('lidar_gt_top_p128_v4p5'), LidarSensor)
        self.assertEqual(lidar_sensor.get_sensor_id(), 'lidar_gt_top_p128_v4p5')

        # Load all data
        for frame_index in lidar_sensor.get_frame_index_range():
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'xyz_s'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'xyz_e'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'intensity'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'timestamp_us'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'dynamic_flag'))
            self.assertFalse(lidar_sensor.has_frame_data(frame_index, 'foo'))

            point_count = lidar_sensor.get_frame_data(frame_index, 'xyz_e').shape[0]
            self.assertEqual(
                lidar_sensor.get_frame_data(frame_index, 'xyz_s').shape,
                lidar_sensor.get_frame_data(frame_index, 'xyz_e').shape)
            self.assertEqual(len(lidar_sensor.get_frame_data(frame_index, 'intensity')), point_count)
            self.assertEqual(len(lidar_sensor.get_frame_data(frame_index, 'timestamp_us')), point_count)
            self.assertEqual(len(lidar_sensor.get_frame_data(frame_index, 'dynamic_flag')), point_count)

            self.assertGreater(len(lidar_sensor.get_frame_labels(frame_index)), 0,
                               f'no labels for lidar frame {frame_index}')

    def test_shard_sensor_camera(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(camera_sensor := loader.get_sensor('camera_cross_left_120fov'), CameraSensor)
        self.assertEqual(camera_sensor.get_sensor_id(), 'camera_cross_left_120fov')

        self.assertIsInstance(camera_sensor.get_camera_model_parameters(), FThetaCameraModelParameters)
        self.assertEqual(camera_sensor.get_camera_mask_image().size, (3848, 2168))

        # Decode all camera frames
        for frame_index in camera_sensor.get_frame_index_range():
            self.assertEqual(camera_sensor.get_frame_image(frame_index).size, (3848, 2168))

    def test_shard_sensor(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(sensor := loader.get_sensor('camera_front_wide_120fov'), Sensor)
        self.assertEqual(sensor.get_sensor_id(), 'camera_front_wide_120fov')

        # Check some known values across shards
        reference_T_sensor_rig = np.array(
            [[-0.01677173376083374, -0.006367421709001064, 0.9998390674591064, 1.7747461795806885],
             [-0.9998273849487305, 0.008099998347461224, -0.016719957813620567, 0.0014275670982897282],
             [-0.007992231287062168, -0.9999469518661499, -0.006502173840999603, 1.448835015296936],
             [0.0, 0.0, 0.0, 1.0]])
        self.assertIsNone(np.testing.assert_array_equal(sensor.get_T_sensor_rig(), reference_T_sensor_rig))

        self.assertEqual(sensor.get_frames_count(), 9)
        self.assertEqual(sensor.get_frame_index_range(), range(0, 9, 1))

        # first frame in first shard
        reference_T_rig_world_0_start = np.array(
            [[0.9999999403953552, -0.0002584185858722776, -1.9103787053609267e-05, -0.000112491863546893],
             [0.0002584183239378035, 0.9999999403953552, -1.4615495274483692e-05, 0.0009787323651835322],
             [1.9107563275611028e-05, 1.4610557627747767e-05, 1.0, 0.0002327073598280549], [0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(sensor.get_frames_timestamps_us()[0], 1648597319007668)
        self.assertEqual(sensor.get_closest_frame_index(1648597319007668), 0)
        self.assertEqual(sensor.get_frame_timestamp_us(0, FrameTimepoint.START), 1648597318976057)
        self.assertEqual(sensor.get_frame_timestamp_us(0, FrameTimepoint.END), 1648597319007668)
        self.assertIsNone(
            np.testing.assert_array_equal(sensor.get_frame_T_rig_world(0, FrameTimepoint.START),
                                          reference_T_rig_world_0_start))
        self.assertIsNone(
            np.testing.assert_array_equal(sensor.get_frame_T_sensor_world(0, FrameTimepoint.START),
                                          reference_T_rig_world_0_start @ reference_T_sensor_rig))
        self.assertEqual(sensor.get_frame_image(0).size, (3848, 2168))

        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(0, FrameTimepoint.END),
                [[0.9999999403953552, -0.0003373134240973741, -2.4829752874211408e-05, -0.00016799320292193443],
                 [0.00033731304574757814, 0.9999999403953552, -1.6016723748180084e-05, 0.0013537255581468344],
                 [2.4835153453750536e-05, 1.6008347301976755e-05, 1.0, 0.0003075051645282656], [0.0, 0.0, 0.0, 1.0]]))

        # second frame in first shard
        self.assertEqual(sensor.get_frames_timestamps_us()[1], 1648597319107667)
        self.assertEqual(sensor.get_closest_frame_index(1648597319107667), 1)
        self.assertEqual(sensor.get_frame_timestamp_us(1, FrameTimepoint.START), 1648597319076056)
        self.assertEqual(sensor.get_frame_timestamp_us(1, FrameTimepoint.END), 1648597319107667)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(1, FrameTimepoint.START),
                [[0.9999999403953552, -0.0003129387041553855, -2.207939360232558e-05, -0.0003473191463854164],
                 [0.0003129389660898596, 0.9999999403953552, 1.2848839105572551e-05, 0.001945359050296247],
                 [2.207537181675434e-05, -1.285574853682192e-05, 1.0, 0.00031924343784339726], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(1, FrameTimepoint.END),
                [[0.9999999403953552, -0.0002774616295937449, -2.3144866645452566e-05, -0.0004720762372016907],
                 [0.00027746273553930223, 0.9999999403953552, 4.720890865428373e-05, 0.001981734996661544],
                 [2.3131768102757633e-05, -4.721532968687825e-05, 1.0, 0.00020380942441988736], [0.0, 0.0, 0.0, 1.0]]))

        # first frame in second shard
        self.assertEqual(sensor.get_frames_timestamps_us()[3], 1648597319307655)
        self.assertEqual(sensor.get_closest_frame_index(1648597319307655), 3)
        self.assertEqual(sensor.get_frame_timestamp_us(3, FrameTimepoint.START), 1648597319276044)
        self.assertEqual(sensor.get_frame_timestamp_us(3, FrameTimepoint.END), 1648597319307655)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(3, FrameTimepoint.START),
                [[1.0, -4.873385114478879e-05, 6.169429070723709e-06, -0.0007022547069936991],
                 [4.8732828872743994e-05, 1.0, 0.00016509750275872648, 0.0004695889074355364],
                 [-6.177474460855592e-06, -0.00016509719716850668, 1.0, -0.0007893226575106382], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(3, FrameTimepoint.END),
                [[1.0, -7.978349458426237e-05, 1.87974455911899e-05, -0.0005667057703249156],
                 [7.978126086527482e-05, 1.0, 0.00011895087664015591, 0.0006683521205559373],
                 [-1.88069370778976e-05, -0.00011894937779288739, 1.0, -0.000652956950943917], [0.0, 0.0, 0.0, 1.0]]))

        # second frame in second shard
        self.assertEqual(sensor.get_frames_timestamps_us()[4], 1648597319407642)
        self.assertEqual(sensor.get_closest_frame_index(1648597319407642), 4)
        self.assertEqual(sensor.get_frame_timestamp_us(4, FrameTimepoint.START), 1648597319376031)
        self.assertEqual(sensor.get_frame_timestamp_us(4, FrameTimepoint.END), 1648597319407642)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(4, FrameTimepoint.START),
                [[1.0, -8.582309237681329e-05, -1.9762739611906e-05, -0.0008120734710246325],
                 [8.582702139392495e-05, 1.0, 0.00019869946117978543, 0.0008941364358179271],
                 [1.9745686586247757e-05, -0.00019870116375386715, 1.0, -0.0007007886306382716], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(4, FrameTimepoint.END),
                [[1.0, -8.811573934508488e-05, -3.058437869185582e-05, -0.000894361874088645],
                 [8.812233136268333e-05, 1.0, 0.00021557316358666867, 0.0009795179357752204],
                 [3.0565381166525185e-05, -0.00021557585569098592, 1.0, -0.0006927505019120872], [0.0, 0.0, 0.0, 1.0]]))

        # first frame in third shard
        self.assertEqual(sensor.get_frames_timestamps_us()[6], 1648597319607634)
        self.assertEqual(sensor.get_closest_frame_index(1648597319607634), 6)
        self.assertEqual(sensor.get_frame_timestamp_us(6, FrameTimepoint.START), 1648597319576023)
        self.assertEqual(sensor.get_frame_timestamp_us(6, FrameTimepoint.END), 1648597319607634)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(6, FrameTimepoint.START),
                [[1.0, 5.302716817823239e-05, -3.7100351619301364e-05, -0.0008849087171256542],
                 [-5.301716009853408e-05, 0.9999999403953552, 0.00026968304882757366, 0.00025372987147420645],
                 [3.711465251399204e-05, -0.0002696810697671026, 0.9999999403953552, -0.0010027275420725346],
                 [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(6, FrameTimepoint.END),
                [[1.0, 6.46427579340525e-05, -4.565540803014301e-05, -0.0008179104188457131],
                 [-6.462900637416169e-05, 0.9999999403953552, 0.00030105686164461076, 0.00011379005445633084],
                 [4.5674863940803334e-05, -0.0003010538930539042, 0.9999999403953552, -0.001094557112082839],
                 [0.0, 0.0, 0.0, 1.0]]))

        # second frame in third shard
        self.assertEqual(sensor.get_frames_timestamps_us()[7], 1648597319707641)
        self.assertEqual(sensor.get_closest_frame_index(1648597319707641), 7)
        self.assertEqual(sensor.get_frame_timestamp_us(7, FrameTimepoint.START), 1648597319676030)
        self.assertEqual(sensor.get_frame_timestamp_us(7, FrameTimepoint.END), 1648597319707641)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(7, FrameTimepoint.START),
                [[1.0, -0.00023531311308033764, -1.2050433724652976e-05, -0.0001188056412502192],
                 [0.00023531373881269246, 1.0, 5.182733366382308e-05, 0.001120504573918879],
                 [1.2038238310196903e-05, -5.183016764931381e-05, 1.0, -0.00019744792371056974], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(7, FrameTimepoint.END),
                [[0.9999999403953552, -0.000326584093272686, -1.8247705213525478e-07, 8.249922393588349e-05],
                 [0.000326584093272686, 0.9999999403953552, -2.2472264390671626e-05, 0.001441847998648882],
                 [1.8981612015522842e-07, 2.2472202545031905e-05, 1.0, 4.710151915787719e-05], [0.0, 0.0, 0.0, 1.0]]))
