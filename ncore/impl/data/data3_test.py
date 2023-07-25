# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import unittest
import random
import itertools

from pathlib import Path

import numpy as np
import parameterized

from .data3 import ShardDataLoader, Sensor, CameraSensor, LidarSensor
from .types import FrameTimepoint, FThetaCameraModelParameters


class TestData3Loader(unittest.TestCase):
    ''' Test to verify functionality of V3 data loader '''
    def setUp(self):
        self.random = random.Random(x=0)  # seed deterministically
        self.all_shards = sorted([str(p) for p in Path('external/test-data-v3-shards').iterdir() if p.match('*.itar')])

    @parameterized.parameterized.expand(itertools.product((False, True), (False, True)))
    def test_shard_loader(self, open_consolidated: bool, reload_store_resources: bool):
        shard_num_poses = [5, 5, 3]
        self.assertEqual(len(self.all_shards), 3)

        def check(start, end):
            # Randomize shard path oder
            local_shards = self.all_shards[start:end]
            self.random.shuffle(local_shards)

            loader = ShardDataLoader(local_shards, open_consolidated=open_consolidated)

            if reload_store_resources:
                loader.reload_store_resources()

            # expected number of total poses is sum of per-shard poses minus duplicated/removed poses at shard boundaries
            expected_num_poses = sum(shard_num_poses[start:end]) - (end - start - 1)

            self.assertEqual(len(loader.get_camera_ids()), 10)
            self.assertEqual(len(loader.get_lidar_ids()), 1)
            self.assertEqual(len(loader.get_radar_ids()), 0)
            self.assertEqual(len(loader.get_sensor_ids()), 11)

            poses = loader.get_poses()
            self.assertEqual(poses.T_rig_world_base.shape, (4, 4))
            self.assertEqual(poses.T_rig_world_timestamps_us.shape, (expected_num_poses, ))
            self.assertEqual(poses.T_rig_worlds.shape, (expected_num_poses, 4, 4))

            for local_shard_idx, shard_id in enumerate(range(start, end)):
                # check *single* shard sub-range pose lookup
                self.assertEqual(
                    loader.get_poses(local_shard_idx, local_shard_idx + 1).T_rig_world_timestamps_us.shape,
                    (shard_num_poses[shard_id], ))

            self.assertEqual(loader.get_sequence_id(with_shard_range=False), 'c9b05cf4-afb9-11ec-b3c2-00044bf65fcb')
            self.assertEqual(
                loader.get_sequence_id(with_shard_range=True),
                'c9b05cf4-afb9-11ec-b3c2-00044bf65fcb_' + '_'.join([str(shard_id) for shard_id in range(start, end)]))

            self.assertEqual(loader.get_calibration_type(), 'scene-calib')
            self.assertEqual(loader.get_egomotion_type(), 'lidar-egomotion')

            # make sure returned paths are absolute and ordered by shard-id
            self.assertEqual(loader.get_shard_paths(), [str(Path(p).absolute()) for p in self.all_shards[start:end]])
            self.assertEqual(loader.get_shard_ids(), list(range(start, end)))

            # check tracks API
            tracks, track_properties = loader.get_tracks()
            self.assertEqual(len(tracks.track_labels), 21)
            self.assertEqual(track_properties, None)  # exported data didn't contain track-properties

        # check all shard slice variants
        for end in range(1, len(self.all_shards) + 1):
            for start in range(0, end):
                check(start, end)

    def test_shard_sensor_lidar(self):
        self.assertEqual(len(self.all_shards), 3)
        loader = ShardDataLoader(self.all_shards)

        self.assertIsInstance(lidar_sensor := loader.get_sensor('lidar_gt_top_p128_v4p5'), LidarSensor)
        self.assertEqual(lidar_sensor.get_sensor_id(), 'lidar_gt_top_p128_v4p5')
        self.assertEqual(lidar_sensor.get_sensor_id(),
                         loader.get_lidar_sensor('lidar_gt_top_p128_v4p5').get_sensor_id())

        # Load all data
        for frame_index in lidar_sensor.get_frame_index_range():
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'xyz_s'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'xyz_e'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'intensity'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'timestamp_us'))
            self.assertTrue(lidar_sensor.has_frame_data(frame_index, 'dynamic_flag'))
            self.assertFalse(lidar_sensor.has_frame_data(frame_index, 'foo'))
            self.assertFalse(lidar_sensor.has_frame_data(
                frame_index, 'semantic_class'))  # the current NV test dataset doesn't have semantic_class properties

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
        self.assertEqual(camera_sensor.get_sensor_id(),
                         loader.get_camera_sensor('camera_cross_left_120fov').get_sensor_id())

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
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32)
        self.assertIsNone(np.testing.assert_array_equal(sensor.get_T_sensor_rig(), reference_T_sensor_rig))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(sensor.get_T_rig_sensor(), np.linalg.inv(reference_T_sensor_rig)))

        self.assertEqual(sensor.get_frames_count(), 10)

        self.assertEqual(sensor.get_frames_count(0, 1), 4)
        self.assertEqual(sensor.get_frames_count(1, 2), 4)
        self.assertEqual(sensor.get_frames_count(2, 3), 2)
        self.assertEqual(sensor.get_frames_count(0, 2), 8)

        self.assertEqual(len(sensor.get_frames_timestamps_us(0, 1)), 4)
        self.assertEqual(len(sensor.get_frames_timestamps_us(1, 2)), 4)
        self.assertEqual(len(sensor.get_frames_timestamps_us(2, 3)), 2)
        self.assertEqual(len(sensor.get_frames_timestamps_us(0, 2)), 8)

        self.assertEqual(sensor.get_frame_index_range(), range(0, 10, 1))

        # Check that all sensor timestamps are strictly monotonically increasing
        self.assertTrue(np.all((timestamps := sensor.get_frames_timestamps_us())[:-1] < timestamps[1:]))

        # first frame in first shard
        reference_T_rig_world_0_start = np.array(
            [[1.0, -9.436352835589197e-21, -8.560157409657141e-20, -2.168404344971009e-19],
             [9.436352835589197e-21, 1.0, 1.0152847796453393e-19, -2.168404344971009e-19],
             [8.560157409657141e-20, -1.0152847796453393e-19, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32)
        self.assertEqual(sensor.get_frames_timestamps_us()[0], 1648597318907676)
        self.assertEqual(sensor.get_closest_frame_index(1648597318907676), 0)
        self.assertEqual(sensor.get_frame_timestamp_us(0, FrameTimepoint.START), 1648597318900083)
        self.assertEqual(sensor.get_frame_timestamp_us(0, FrameTimepoint.END), 1648597318907676)
        self.assertIsNone(
            np.testing.assert_array_equal(sensor.get_frame_T_rig_world(0, FrameTimepoint.START),
                                          reference_T_rig_world_0_start))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(sensor.get_frame_T_world_rig(0, FrameTimepoint.START),
                                                 np.linalg.inv(reference_T_rig_world_0_start)))
        self.assertIsNone(
            np.testing.assert_array_equal(sensor.get_frame_T_sensor_world(0, FrameTimepoint.START),
                                          reference_T_rig_world_0_start @ reference_T_sensor_rig))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(sensor.get_frame_T_world_sensor(0, FrameTimepoint.START),
                                                 np.linalg.inv(reference_T_rig_world_0_start @ reference_T_sensor_rig)))
        self.assertEqual(sensor.get_frame_image(0).size, (3848, 2168))

        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(0, FrameTimepoint.END),
                [[1.0, -2.5826881028478965e-05, -1.9094422896159813e-06, -1.1242672371736262e-05],
                 [2.5826877390500158e-05, 1.0, -1.4604810303353588e-06, 9.78165480773896e-05],
                 [1.9094798062724294e-06, 1.4604316902477876e-06, 1.0, 2.3257258362718858e-05], [0.0, 0.0, 0.0, 1.0]]))

        # second frame in first shard
        self.assertEqual(sensor.get_frames_timestamps_us()[1], 1648597319007668)
        self.assertEqual(sensor.get_closest_frame_index(1648597319007668), 1)
        self.assertEqual(sensor.get_frame_timestamp_us(1, FrameTimepoint.START), 1648597318976057)
        self.assertEqual(sensor.get_frame_timestamp_us(1, FrameTimepoint.END), 1648597319007668)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(1, FrameTimepoint.START),
                [[0.9999999403953552, -0.0002584185858722776, -1.9103787053609267e-05, -0.00011249187082285061],
                 [0.0002584183239378035, 0.9999999403953552, -1.4615495274483692e-05, 0.0009787322487682104],
                 [1.9107563275611028e-05, 1.4610557627747767e-05, 1.0, 0.00023270734527613968], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(1, FrameTimepoint.END),
                [[0.9999999403953552, -0.0003373134240973741, -2.4829752874211408e-05, -0.00016799321747384965],
                 [0.00033731304574757814, 0.9999999403953552, -1.6016723748180084e-05, 0.0013537255581468344],
                 [2.4835153453750536e-05, 1.6008347301976755e-05, 1.0, 0.0003075051645282656], [0.0, 0.0, 0.0, 1.0]]))

        # first frame in second shard
        self.assertEqual(sensor.get_frames_timestamps_us()[4], 1648597319307655)
        self.assertEqual(sensor.get_closest_frame_index(1648597319307655), 4)
        self.assertEqual(sensor.get_frame_timestamp_us(4, FrameTimepoint.START), 1648597319276044)
        self.assertEqual(sensor.get_frame_timestamp_us(4, FrameTimepoint.END), 1648597319307655)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(4, FrameTimepoint.START),
                [[1.0, -4.873385114478879e-05, 6.169429070723709e-06, -0.00070225476520136],
                 [4.8732828872743994e-05, 1.0, 0.00016509750275872648, 0.0004695889074355364],
                 [-6.177474460855592e-06, -0.00016509719716850668, 1.0, -0.0007893225993029773], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(4, FrameTimepoint.END),
                [[1.0, -7.978349458426237e-05, 1.87974455911899e-05, -0.0005667058285325766],
                 [7.978126086527482e-05, 1.0, 0.00011895087664015591, 0.0006683520623482764],
                 [-1.88069370778976e-05, -0.00011894937779288739, 1.0, -0.0006529568927362561], [0.0, 0.0, 0.0, 1.0]]))

        # second frame in second shard
        self.assertEqual(sensor.get_frames_timestamps_us()[5], 1648597319407642)
        self.assertEqual(sensor.get_closest_frame_index(1648597319407642), 5)
        self.assertEqual(sensor.get_frame_timestamp_us(5, FrameTimepoint.START), 1648597319376031)
        self.assertEqual(sensor.get_frame_timestamp_us(5, FrameTimepoint.END), 1648597319407642)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(5, FrameTimepoint.START),
                [[1.0, -8.582309237681329e-05, -1.9762739611906e-05, -0.0008120735292322934],
                 [8.582702139392495e-05, 1.0, 0.00019869946117978543, 0.0008941364358179271],
                 [1.9745686586247757e-05, -0.00019870116375386715, 1.0, -0.0007007886306382716], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(5, FrameTimepoint.END),
                [[1.0, -8.811573934508488e-05, -3.058437869185582e-05, -0.0008943619322963059],
                 [8.812233136268333e-05, 1.0, 0.00021557316358666867, 0.0009795178193598986],
                 [3.0565381166525185e-05, -0.00021557585569098592, 1.0, -0.0006927504437044263], [0.0, 0.0, 0.0, 1.0]]))

        # first frame in third shard
        self.assertEqual(sensor.get_frames_timestamps_us()[8], 1648597319707641)
        self.assertEqual(sensor.get_closest_frame_index(1648597319707641), 8)
        self.assertEqual(sensor.get_frame_timestamp_us(8, FrameTimepoint.START), 1648597319676030)
        self.assertEqual(sensor.get_frame_timestamp_us(8, FrameTimepoint.END), 1648597319707641)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(8, FrameTimepoint.START),
                [[1.0, -0.00023531311308033764, -1.2050433724652976e-05, -0.00011880564852617681],
                 [0.00023531373881269246, 1.0, 5.182733366382308e-05, 0.001120504573918879],
                 [1.2038238310196903e-05, -5.183016764931381e-05, 1.0, -0.0001974479091586545], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(8, FrameTimepoint.END),
                [[0.9999999403953552, -0.000326584093272686, -1.824770663461095e-07, 8.24992312118411e-05],
                 [0.000326584093272686, 0.9999999403953552, -2.2472264390671626e-05, 0.001441847998648882],
                 [1.8981614857693785e-07, 2.2472202545031905e-05, 1.0, 4.7101515519898385e-05], [0.0, 0.0, 0.0, 1.0]]))

        # second frame in third shard
        self.assertEqual(sensor.get_frames_timestamps_us()[9], 1648597319807616)
        self.assertEqual(sensor.get_closest_frame_index(1648597319807616), 9)
        self.assertEqual(sensor.get_frame_timestamp_us(9, FrameTimepoint.START), 1648597319776005)
        self.assertEqual(sensor.get_frame_timestamp_us(9, FrameTimepoint.END), 1648597319807616)
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(9, FrameTimepoint.START),
                [[1.0, -0.0002038357633864507, 5.69873805034149e-07, -0.0003055224078707397],
                 [0.0002038357051787898, 1.0, 9.321869583800435e-05, 0.0011638260912150145],
                 [-5.888750820304267e-07, -9.321857214672491e-05, 1.0, -0.0005736812017858028], [0.0, 0.0, 0.0, 1.0]]))
        self.assertIsNone(
            np.testing.assert_array_equal(
                sensor.get_frame_T_rig_world(9, FrameTimepoint.END),
                [[1.0, -0.00016944203525781631, -2.284358970428002e-06, -0.0004524229443632066],
                 [0.00016944236995186657, 1.0, 0.0001480117643950507, 0.0011142912553623319],
                 [2.2592796540266136e-06, -0.00014801214274484664, 1.0, -0.0008234084234572947], [0.0, 0.0, 0.0, 1.0]]))
