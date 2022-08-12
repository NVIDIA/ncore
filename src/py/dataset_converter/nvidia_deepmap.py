# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import os
import glob
import json
import cv2
import numpy as np

from google.protobuf import text_format
from pyarrow.parquet import ParquetDataset
from collections import defaultdict
from protobuf_to_dict import protobuf_to_dict

from src.protos.deepmap import track_data_pb2, pointcloud_pb2
from src.py.dataset_converter import BaseNvidiaDataConverter
from src.py.common.common import (PoseInterpolator, save_pkl, load_pkl, save_pc_dat, is_within_3d_bbox)
from src.cpp.av_utils import unwind_lidar
from src.py.common.nvidia_utils import (compute_ftheta_parameters, extract_pose, extract_sensor_2_sdc,
                              parse_rig_sensors_from_dict, sensor_to_rig, camera_intrinsic_parameters, compute_fw_polynomial,
                              camera_car_mask, vehicle_bbox, LabelParser)


class NvidiaDeepMapConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data converter (based on DeepMap tracks)
    """

    def __init__(self, config):
        super().__init__(config)

        self.sequence_pathnames = sorted(glob.glob(os.path.join(self.root_dir, '*/')))


    def convert_one(self, sequence_path):
        """
        Runs the conversion of a single sequence
        
        Args:
            sequence_path (string): path to the raw sequence data
        
        Return:
            sub_sequence_names List[string]: names of the processed sub-sequences
        """

        self.sequence_name = sequence_path.split(os.sep)[-2]

        sequence_tracks = sorted(glob.glob(os.path.join(sequence_path,'tracks','*/')))

        sub_sequence_names = []
        for track in sequence_tracks:
            self.track_name = track.split(os.sep)[-2]

            # create all the folders
            self.create_folders(os.path.join(self.sequence_name, self.track_name))

            # Initialize the pose variables. Poses can be coupled to either images or lidar frames
            self.poses = []
            self.poses_timestamps = []
            self.lidar_timestamps = []
            self.lidar_data_paths = []
            annotations = {}
            annotations['3d_labels'] = defaultdict(dict)
            frame_annotations = defaultdict(dict)

            # Read rig json file
            with open(os.path.join(sequence_path, 'rig.json'), 'r') as fp:
                self.rig = json.load(fp)

            # Initialize the track aligned track record structure
            self.track_data = track_data_pb2.AlignedTrackRecords()

            # Read in the track record data from a proto file
            # This includes camera_records and lidar_records (see track_record proto for more detail)
            with open(os.path.join(track, 'aligned_track_records.pb.txt'), 'r') as f:
                text_format.Parse(f.read(), self.track_data)

            # Extract all the lidar paths, timestamps and poses from the track record
            self.track_data = protobuf_to_dict(self.track_data)

            self.decode_poses_timestamps()

            self.decode_labels(sequence_path, annotations, frame_annotations)

            self.decode_lidar(sequence_path)

            self.decode_images(sequence_path)

            sub_sequence_names.append(os.path.join(self.sequence_name, self.track_name))

        return sub_sequence_names

    def decode_poses_timestamps(self):
        # Extract poses and timestamps, which are converted to the nvidia convention
        if 'lidar_records' in self.track_data:
            for frame in self.track_data['lidar_records'][0]['records']:
                self.lidar_timestamps.append(frame['timestamp_microseconds'])
                self.lidar_data_paths.append(frame['file_path'])

                if 'pose' in frame:
                    self.poses_timestamps.append(frame['timestamp_microseconds'])
                    self.poses.append(extract_pose(frame['pose']))

        if 'camera_records' in self.track_data:
            for frame in self.track_data['camera_records'][0]['records']:
                if 'pose' in frame:
                    self.poses_timestamps.append(frame['timestamp_microseconds'])
                    self.poses.append(extract_pose(frame['pose']))


        # Stack and sort the poses
        self.poses = np.stack(self.poses)
        self.poses_timestamps = np.stack(self.poses_timestamps).astype(np.float64)
        sort_idx = np.argsort(self.poses_timestamps)

        # All the available poses
        self.poses = self.poses[sort_idx]
        self.poses_timestamps = self.poses_timestamps[sort_idx]

        if os.path.exists(os.path.join(self.output_dir, self.sequence_name, 'base_pose.npz')):
            self.base_pose = np.load(os.path.join(self.output_dir, self.sequence_name, 'base_pose.npz'))['base_pose']
        else:
            self.base_pose = self.poses[0]
            np.savez(os.path.join(self.output_dir, self.sequence_name, 'base_pose.npz'), base_pose=self.base_pose)

        # Convert the poses to the sequence coordinate frame
        self.poses = np.linalg.inv(self.base_pose) @ self.poses

        # Save the poses
        poses_save_path = os.path.join(self.output_dir, self.sequence_name, self.track_name,
                        self.poses_save_dir, 'poses.npz')

        np.savez(poses_save_path, base_pose=self.base_pose, ego_poses=self.poses, timestamps=self.poses_timestamps)

    def decode_images(self, sequence_path):
        # Parse the rig calibration file
        calibration_data = parse_rig_sensors_from_dict(self.rig)
        camera_timestamps = defaultdict(list)

        # Filter the images based on the pose timestamps
        for camera in self.CAMERA_2_IDTYPERIG.keys():
            cam_id, cam_type, cam_id_rig = self.CAMERA_2_IDTYPERIG[camera]

            # Target folder for all camera-specific outputs
            camera_base_save_path = os.path.join(self.output_dir, self.sequence_name, self.track_name, self.image_save_dir, 'image_' + cam_id)

            # Get the camera timestamps
            frame_timestamps = np.genfromtxt(os.path.join(sequence_path, 'camera_data/', camera + '.mp4.timestamps'), delimiter='\t', dtype=int)

            # Get the frame index of the first and last frame
            start_idx = np.where(frame_timestamps[:,1] > self.poses_timestamps[0] + self.CAM2ROLLINGSHUTTERDELAY[cam_type] + self.CAM2EXPOSURETIME[cam_type])[0][0]
            end_idx = np.where(frame_timestamps[:,1] >= self.poses_timestamps[-1])[0][0]

            frame_timestamps = frame_timestamps[start_idx:end_idx, :]

            # Extract all the images
            vidcap = cv2.VideoCapture(os.path.join(sequence_path, 'camera_data/', camera + '.mp4'))
            success, image = vidcap.read()
            count = 0
            save_frame = 0
            img_height, img_width,  = image.shape[0:2]
            while success:
                if frame_timestamps[0,0] <= count <= frame_timestamps[-1,0]:
                    save_path = os.path.join(camera_base_save_path, str(save_frame).zfill(self.INDEX_DIGITS) + '.jpeg')
                    cv2.imwrite(save_path, image)     # save frame as JPEG file
                    save_frame += 1

                if count > frame_timestamps[-1,0]:
                    break
                success,image = vidcap.read()
                count += 1

            # Extract the metadata (get the relative transformation to the lidar sensor as the rig might change
            T_cam_rig = sensor_to_rig(calibration_data[cam_id_rig])
            T_lidar_rig = sensor_to_rig(calibration_data[self.LIDAR_SENSORNAME])
            lidar_calib_path = os.path.join(sequence_path, 'to_vehicle_transform_lidar00.pb.txt')
            T_lidar_sdc = extract_sensor_2_sdc(lidar_calib_path)

            # Recompute T_cam_rig
            T_cam_rig = T_lidar_sdc @ np.linalg.inv(T_lidar_rig) @ T_cam_rig

            intrinsic = camera_intrinsic_parameters(calibration_data[cam_id_rig])

            # Estimate the forward polynomial and other F-theta parameters
            fw_poly_coeff = compute_fw_polynomial(intrinsic)
            max_ray_distortion, max_angle = compute_ftheta_parameters(np.concatenate((intrinsic, fw_poly_coeff)))
            intrinsic =  np.concatenate((intrinsic, fw_poly_coeff, max_ray_distortion, max_angle))

            cam_pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

            # Constant mask image, which currently only contains the ego car mask
            # TODO: extend this with dynamic object masks
            mask_image = camera_car_mask(calibration_data[cam_id_rig])

            for frame_idx, frame in enumerate(frame_timestamps):
                mask_image.get_image().save(os.path.join(camera_base_save_path, f'mask_{str(frame_idx).zfill(self.INDEX_DIGITS)}.png'), optimize=True)

                metadata = {}
                metadata['img_width'] = img_width
                metadata['img_height'] = img_height
                metadata['rolling_shutter_direction'] = 1 # 1 = TOP_TO_BOTTOM, 2 = LEFT_TO_RIGHT, 3 = BOTTOM_TO_TOP, 4 = RIGHT_TO_LEFT
                metadata['camera_model'] = 'f_theta' if cam_type in ['wide', 'fisheye'] else 'pinhole'
                metadata['exposure_time'] = self.CAM2EXPOSURETIME[cam_type]
                metadata['intrinsic'] = intrinsic
                metadata['T_cam_rig'] = T_cam_rig

                # Interpolate the start and end pose to the timestamps of the first and last row
                sofTimestamp = frame[1] - self.CAM2ROLLINGSHUTTERDELAY[cam_type] - self.CAM2EXPOSURETIME[cam_type] / 2
                eofTimestamp = frame[1] - self.CAM2EXPOSURETIME[cam_type] / 2
                metadata['ego_pose_timestamps'] = np.array([sofTimestamp, eofTimestamp])
                metadata['ego_pose_s'] = cam_pose_interpolator.interpolate_to_timestamps(sofTimestamp)[0]
                metadata['ego_pose_e'] = cam_pose_interpolator.interpolate_to_timestamps(eofTimestamp)[0]

                metadata_save_path = os.path.join(camera_base_save_path, str(frame_idx).zfill(self.INDEX_DIGITS) + '.pkl')

                save_pkl(metadata, metadata_save_path)

                # Save the camera pose timestamps, corresponds approximately to the timestamp of the principle point pixel
                camera_timestamps[cam_id].append((eofTimestamp + sofTimestamp) / 2)

        # Save all camera timestamps
        for cam in camera_timestamps.keys():
            camera_timestamps[cam] = np.stack(camera_timestamps[cam])

        image_t_save_path =  os.path.join(self.output_dir, self.sequence_name, self.track_name,
                        self.image_save_dir, 'timestamps.pkl')

        save_pkl(camera_timestamps, image_t_save_path)

    def decode_labels(self, sequence_path, annotations, frame_annotations):

        # Check if the labels for this sequence were already extracted (done only once, not for each track)
        if not os.path.exists(os.path.join(self.output_dir, self.sequence_name, 'frame_labels.pkl')):
            # Read the pandas file
            dataset = ParquetDataset(os.path.join(sequence_path, 'labels', 'autolabels.parquet'))
            table = dataset.read()
            label_data = table.to_pandas()
            label_data = label_data.reset_index()  # make sure indexes pair with number of rows

            for _, row in label_data.iterrows():
                if row['label_name'] in LabelParser.LABEL_STRING_TO_LABEL_ID.keys():

                    track_id = row['trackline_id']
                    label_timestamp = row['detection_timestamp']

                    cuboid = np.array([row['centroid_x'], row['centroid_y'], row['centroid_z'],
                                       row['dim_x'], row['dim_y'], row['dim_z'],
                                       row['rot_x'], row['rot_y'], row['rot_z']], dtype=np.float32)

                    if label_timestamp not in frame_annotations:
                        frame_annotations[label_timestamp]['lidar_labels'] = []

                    frame_annotations[label_timestamp]['lidar_labels'].append({'id': len(frame_annotations[label_timestamp]['lidar_labels']),
                                                                                'name': track_id,
                                                                                'label': LabelParser.LABEL_STRING_TO_LABEL_ID[row['label_name']],
                                                                                '3D_bbox': cuboid,
                                                                                'num_points':
                                                                                    -1,
                                                                                'detection_difficulty_level':
                                                                                    -1,
                                                                                'combined_difficulty_level':
                                                                                    -1,
                                                                                'global_speed':
                                                                                    -1,
                                                                                'global_accel':
                                                                                    -1})


                    if track_id not in annotations['3d_labels']:
                        annotations['3d_labels'][track_id]['dynamic_flag'] = 1 if LabelParser.LABEL_STRING_TO_LABEL_ID[row['label_name']] in LabelParser.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC else 0
                        annotations['3d_labels'][track_id]['type'] = LabelParser.LABEL_STRING_TO_LABEL_ID[row['label_name']]
                        annotations['3d_labels'][track_id]['lidar'] = {}


                    annotations['3d_labels'][track_id]['lidar'][label_timestamp] = {'3D_bbox': cuboid,
                                                                        'num_point': -1,
                                                                        'global_speed': -1,
                                                                        'global_accel': -1,}

                    # TODO: check if this user-defined threshold makes sense
                    if LabelParser.LABEL_STRING_TO_LABEL_ID[row['label_name']] not in LabelParser.LABEL_STRINGS_UNCONDITIONALLY_STATIC and np.linalg.norm([row['velocity_x'], row['velocity_y']]) >= 1/3.6:
                        annotations['3d_labels'][track_id]['dynamic_flag'] = 1

            # Save the accumulated data
            labels_save_path =  os.path.join(self.output_dir, self.sequence_name, 'labels.pkl')
            save_pkl(annotations, labels_save_path)

            # Save the per frame data
            frame_labels_save_path =  os.path.join(self.output_dir, self.sequence_name, 'frame_labels.pkl')
            save_pkl(frame_annotations, frame_labels_save_path)

    def decode_lidar(self, sequence_path):
        annotations  = load_pkl(os.path.join(self.output_dir, self.sequence_name, 'labels.pkl'))

        frame_annotations  = load_pkl(os.path.join(self.output_dir, self.sequence_name, 'frame_labels.pkl'))

        fa_timestamps = np.array(sorted(list(frame_annotations.keys())))

        lidar_calib_path = os.path.join(sequence_path, 'to_vehicle_transform_lidar00.pb.txt')
        T_lidar_rig = extract_sensor_2_sdc(lidar_calib_path)

        # Load vehicle bounding box (defined in rig frame)
        vehicle_bbox_rig = vehicle_bbox(self.rig)
        vehicle_bbox_rig[3:6] += self.LIDAR_FILTER_VEHICLE_BBOX_PADDING  # pad the bounding box slightly

        # Initialize the pose interpolator object
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

        lidar_end_timestmap = []
        # We remove the first and the last lidar frame such that the poses do not have to be extrapolated
        frame_idx = 0
        for frame_path in self.lidar_data_paths:

            # First frame might not work as the pose is available only at the middle of the frame
            try:
                # Load the point cloud data
                data = pointcloud_pb2.PointCloud()
                with open(os.path.join(sequence_path, 'tracks', frame_path), 'rb') as f:
                    data.ParseFromString(f.read())

                raw_pc = np.concatenate([np.array(data.data.points_x)[:,None],
                                         np.array(data.data.points_y)[:,None],
                                         np.array(data.data.points_z)[:,None]], axis=1)

                # spherical_coordinates = euclidean_2_spherical_coords(raw_pc)
                intensities = np.frombuffer(data.data.intensities, dtype=np.uint8)

                # Save the end time stamp of the lidar spin
                lidar_end_timestmap.append(data.meta_data.end_timestamp_microseconds)

                # Find the closest frame in the annotations
                time_diff = np.abs(fa_timestamps - (data.meta_data.end_timestamp_microseconds))
                annotation_frame_idx = np.argmin(time_diff)

                if time_diff[annotation_frame_idx] > 10000:
                    print("no corresponding frame found")
                    continue

                # Transform points to rig frame to perform filtering
                raw_pc_homogeneous = np.row_stack([raw_pc.transpose(), np.ones(raw_pc.shape[0], dtype=np.float32)])  # 4 x N
                raw_pc_homogeneous_rig = T_lidar_rig @ raw_pc_homogeneous  # 4 x N

                # Filter out points that are more than LIDAR_FILTER_MIN_RIG_HEIGHT bellow ground (there are some spurious measurements there)
                valid_idx_z = raw_pc_homogeneous_rig[2, :] > self.LIDAR_FILTER_MIN_RIG_HEIGHT

                # Filter outs points that are inside the vehicles bounding-box
                valid_idxs_vehicle_bbox = np.logical_not(is_within_3d_bbox(raw_pc_homogeneous_rig[0:3, :].transpose(), vehicle_bbox_rig))

                # Determine per-column rig-to-world pose and compute per-column lidar-to-world transformations
                column_timestamps = np.array(data.data.column_timestamps_microseconds)
                column_poses = pose_interpolator.interpolate_to_timestamps(column_timestamps)
                T_column_lidar_worlds = column_poses @ T_lidar_rig[None,:,:]

                # Perform per-column unwinding, transforming from lidar to world coordinates
                transformed_pc = unwind_lidar(raw_pc, T_column_lidar_worlds.reshape(-1,4), np.array(data.data.column_indices).reshape(-1,1))

                # Filter points based on distances
                dist = np.linalg.norm(transformed_pc[:,:3] - transformed_pc[:,3:6], axis=1)
                transformed_pc = np.concatenate([transformed_pc, dist[:,None], intensities[:, None], -1*np.ones_like(dist[:,None])], axis=1)

                # Filter points on the distances LIDAR_FILTER_MAX_DISTANCE (remove points that are very far away)
                valid_idx_dist = np.less_equal(dist, self.LIDAR_FILTER_MAX_DISTANCE)
                valid_idx = np.logical_and(np.logical_and(valid_idx_z, valid_idxs_vehicle_bbox), valid_idx_dist)

                # 3D rays in space with accompanying metadata.
                # Format; x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic flag
                # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
                raw_pc = raw_pc[valid_idx,:]
                transformed_pc = transformed_pc[valid_idx,:]

                # Save the per frame label
                anno_save_path =  os.path.join(self.output_dir, self.sequence_name, self.track_name,
                        self.label_save_dir, str(frame_idx).zfill(self.INDEX_DIGITS) + '.pkl')

                save_pkl(frame_annotations[fa_timestamps[annotation_frame_idx]], anno_save_path)

                # Use the bounding boxes to remove dynamic objects
                dynamic_flag = np.zeros_like(transformed_pc[:,0])
                for label in frame_annotations[fa_timestamps[annotation_frame_idx]]['lidar_labels']:
                    label_id = label['name']
                    dynamic_state = annotations['3d_labels'][label_id]['dynamic_flag']
                    # If the object is dynamic update the points that fall in that bounding box
                    if dynamic_state:
                        bbox = label['3D_bbox']
                        bbox[3:6] += self.LIDAR_DYNAMIC_FLAG_BBOX_PADDING # enlarge the bounding box
                        dynamic_flag[is_within_3d_bbox(raw_pc, bbox)] = 1

                transformed_pc[:,-1] = dynamic_flag

                # Use the dynamic masks to remove objects
                # dynamic_flag = []
                # with open(os.path.join(sequence_path, 'tracks', frame_path.replace('lidar_00', 'lidar_00_dynamic_point_masks').replace('.ppb','.pb')), 'rb') as f:
                #     while True:
                #         buf = f.read(8)
                #         if not buf:
                #             break
                #         dynamic_flag.append(np.unpackbits(np.array(struct.unpack('<Q', buf), dtype=np.uint8), bitorder='little'))
                #     byte_array = np.fromfile(f, np.dtype('B'))
                #     dynamic_flag = np.unpackbits(byte_array, bitorder="little")

                # dynamic_flag = np.stack(dynamic_flag).flatten()

                # dynamic_flag = dynamic_flag[:valid_idx.shape[0]]
                # transformed_pc[:,-1] = dynamic_flag[valid_idx]

                lidar_save_path = os.path.join(self.output_dir, self.sequence_name, self.track_name, self.point_cloud_save_dir, str(frame_idx).zfill(self.INDEX_DIGITS) + '.dat.xz')
                save_pc_dat(lidar_save_path, transformed_pc.astype(np.float32))

                # Store metadata of the lidar frame
                metadata = {}
                metadata['T_lidar_rig'] = T_lidar_rig       # Lidar extrinsic parameters (note: this can be assumed to be constant and could be stored only once)
                metadata['T_rig_world'] = column_poses[-1]  # Pose of the rig at the end of the lidar spin, can be used to transform points into a local coordinate frame
                metadata['elevation_angles'] = None         # [TODO: currently missing for NV sensors] Lidar elevation angles, can be used to simulate the lidar or recover points that did not return
                save_pkl(metadata, lidar_save_path.replace('.dat.xz','.pkl'))

                frame_idx += 1

            except Exception as e: # work on python 3.x
                print('Lidar frame conversion failed')

        # Save all lidar timestamps
        lidar_timestamp_save_path = os.path.join(self.output_dir, self.sequence_name, self.track_name, self.point_cloud_save_dir, 'timestamps.npz')
        np.savez(lidar_timestamp_save_path, timestamps=lidar_end_timestmap)
