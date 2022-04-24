from collections import defaultdict

from importlib_metadata import metadata
from src.dataset_converter import DataConverter
import tensorflow.compat.v1 as tf        
tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import os 
import struct
from src.waymo_utils import parse_range_image_and_camera_projection, convert_range_image_to_point_cloud, extrapolate_pose_based_on_velocity,\
                            extract_lidar_labels, extract_camera_labels, extract_projected_labels
from src.common import save_pkl
from PIL import Image

class WaymoConverter(DataConverter):    
    def __init__(self, args):

        self.lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        # Label types
        self.type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.CAMERA_2_IDTYPERIG = {1: ['00', 'pinhole', 'camera_front_50fov'],
                                   2: ['01', 'pinhole', 'camera_front_left_50fov'],
                                   3: ['02', 'pinhole', 'camera_front_right_50fov'],
                                   4: ['03', 'pinhole', 'camera_side_left_50fov'],
                                   5: ['04', 'pinhole', 'camera_side_right_50fov']}

        super().__init__(args)

    def convert_one(self, sequence_path): 
        dataset = tf.data.TFRecordDataset(sequence_path, compression_type='')
        
        # Check that all frames in the dataset have the same sequence name
        for frame_idx, data in enumerate(dataset):
            
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame_idx == 0:
                sequence_name = frame.context.name
            if frame.context.name != sequence_name:
                raise SystemExit("NOT ALL FRAMES BELONG TO THE SAME SEQUENCE. ABORTING THE CONVERSION!")

        # Initialize the pose variables. Poses can come from either frame, camera or lidar data
        # poses_timestamps also holds the source information in the second column: 0 - camera, 1 - lidar, 2 - frame
        poses = []
        poses_timestamps = []
        lidar_timestamps = []
        camera_timestamps = defaultdict(list)
        annotations = defaultdict(dict)

        # create all the folders
        self.create_folders(sequence_name)

        for frame_idx, data in enumerate(dataset):

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # Decode the lidar frame
            self.decode_lidar(frame, frame_idx, lidar_timestamps, sequence_name)

            # Decode the image frames
            self.decode_images(frame, poses, poses_timestamps, camera_timestamps, frame_idx, sequence_name)

            # Decode the annotations
            self.decode_labels(frame, frame_idx, annotations)

        # Decode the pose of the current frame and it timestamp
        self.decode_poses_timestamps(frame, poses_timestamps, camera_timestamps, lidar_timestamps, sequence_name)

        # Decode the pose of the current frame and it timestamp
        self.summarize_labels_across_frames(frame, poses_timestamps, camera_timestamps, lidar_timestamps, sequence_name)


    def decode_poses_timestamps(self, poses, poses_timestamps, camera_timestamps, lidar_timestamps, sequence_name):
        # Stack all the poses
        poses = np.stack(poses)
        poses_timestamps = np.stack(poses_timestamps)
        sort_idx = np.argsort(poses_timestamps)

        # All the available poses
        poses = poses[sort_idx]
        poses_timestamps = poses_timestamps[sort_idx]

        # Stack the lidar timestamps
        lidar_timestamps = np.stack(lidar_timestamps)
        lidar_t_save_path =  os.path.join(self.output_dir, sequence_name, 
                        self.point_cloud_save_dir, 'timestamps.npz')
        np.savez(lidar_t_save_path, timestamps=lidar_timestamps)

        # Stack the lidar timestamps
        for cam in camera_timestamps.keys():
            camera_timestamps[cam] = np.stack(camera_timestamps[cam])

        image_t_save_path =  os.path.join(self.output_dir, sequence_name, 
                        self.image_save_dir, 'timestamps.pkl')

        save_pkl(camera_timestamps, image_t_save_path)

    def decode_lidar(self, frame, frame_idx, lidar_timestamps, sequence_name):

        # In the first frame save the lidar extrinsic matrix
        if frame_idx == 0:
            for lidar in frame.context.laser_calibrations:
                if lidar.name == 1: # 1 equals to the top lidar
                    T_lidar_rig = np.array(lidar.extrinsic.transform).reshape(4,4)
                    lidar2rig_save_path = os.path.join(self.output_dir, sequence_name, self.poses_save_dir, 'T_lidar_rig.npz')
                    np.savez(lidar2rig_save_path, T_lidar_rig=T_lidar_rig)


        # Extract lidar data in form of rays in the world coordinate system
        range_images, camera_projections, range_image_top_pose = parse_range_image_and_camera_projection(frame)

        points, _ = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
            keep_polar_features=False,
            return_rays=True
        )

        # Get the timestamp of the START of the lidar spin 
        lidar_timestamps.append(frame.timestamp_micros)

        # 3d points in the world coordinates system represent with start and end point coordinates [n, 8]
        lidar_save_path =  os.path.join(self.output_dir, sequence_name, 
                                        self.point_cloud_save_dir, str(frame_idx).zfill(4) + '.dat')
        points = points[0]
        dist = np.linalg.norm(points[:,3:6] - points[:,:3],axis=1, keepdims=True)
        points = np.concatenate((points[:,3:6], dist, points[:,6:], -1*np.ones_like(dist)),axis=1)
        points_flat = points.flatten()

        with open(lidar_save_path,'wb') as f:
            f.write(struct.pack('>i', points_flat.size))
            f.write(struct.pack('=%sf' % points_flat.size, *points_flat))


    def decode_images(self, frame, poses, poses_timestamps, camera_timestamps, frame_idx, sequence_name):
        images = sorted(frame.images, key=lambda i:i.name)
        
        for image in images:
            # Get the calibration data
            calib = frame.context.camera_calibrations[image.name - 1]

            # Get the SDC car pose 
            # TODO: Check github issues to confirm that this is correct
            poses.append(np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float64), [4, 4])))
            poses_timestamps.append(image.pose_timestamp * 1e6) # Convert the poses to microseconds

            metadata = {}
            metadata['img_width'] = calib.width
            metadata['img_height'] = calib.height            
            metadata['ego_pose_timestamps'] = np.array([image.camera_trigger_time, image.camera_readout_done_time])
            metadata['exposure_time'] = image.shutter
            metadata['rolling_shutter_direction'] = calib.rolling_shutter_direction
            metadata['camera_model'] = self.CAMERA_2_IDTYPERIG[image.name][1]
            metadata['intrinsic'] = np.array(tf.constant(calib.intrinsic, dtype=tf.float64))
            metadata['T_cam_rig'] = np.array(tf.reshape(tf.constant(calib.extrinsic.transform, dtype=tf.float64), [4, 4])) # Camera to sdc

            assert metadata['rolling_shutter_direction'] in [1,2,3,4], "Weird rolling shutter direction, aborting"

            # Velocity and angular velocity of the SDC at camera pose timestamp in global frame.
            T_SDC_global = np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float64), [4, 4]))
            velocity_global = np.array([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z]).reshape(3,1)
            omega_vehicle = np.array([image.velocity.w_x, image.velocity.w_y, image.velocity.w_z]).reshape(3,1)
            omega_global = np.matmul(T_SDC_global[:3,:3], omega_vehicle)

            metadata['ego_pose_s'] = extrapolate_pose_based_on_velocity(T_SDC_global,velocity_global, omega_global, metadata['ego_pose_timestamps'][0] - image.pose_timestamp)
            metadata['ego_pose_e'] = extrapolate_pose_based_on_velocity(T_SDC_global,velocity_global, omega_global, metadata['ego_pose_timestamps'][1] - image.pose_timestamp)
        
            # Save the camera pose timestamps, corresponds approximately to the timestamp of the principle point pixel
            camera_timestamps[self.CAMERA_2_IDTYPERIG[image.name][0]].append(image.pose_timestamp * 1e6)

            # Save the image and its metadata
            im = Image.fromarray(np.array(tf.image.decode_jpeg(image.image)))
    
            img_save_path =  os.path.join(self.output_dir, sequence_name, 
                             self.image_save_dir, 'image_{}'.format(self.CAMERA_2_IDTYPERIG[image.name][0]), 
                             str(frame_idx).zfill(4) + '.jpeg')
            im.save(img_save_path)
    
            save_pkl(metadata, img_save_path.replace('.jpeg','.pkl'))


    def decode_labels(self, frame, frame_idx, annotations):

        # Extract lidar labels 
        lidar_labels = extract_lidar_labels(frame)

        # Extract camera labels (annotated in 2d)
        camera_labels = extract_camera_labels(frame)

        # Extract lidar labels projected to the images
        projected_lidar_labels = extract_projected_labels(frame)

        annotations[frame_idx] = {
                    'lidar_labels': lidar_labels,
                    'camera_labels': camera_labels,
                    'projected_lidar_labels': projected_lidar_labels
        }

    def summarize_labels_across_frames(self,):
        
