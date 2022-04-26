from __future__ import annotations
from collections import defaultdict
from src.dataset_converter import DataConverter
import tensorflow.compat.v1 as tf        
tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import cv2
import os 
import struct
import glob
from src.waymo_utils import parse_range_image_and_camera_projection, convert_range_image_to_point_cloud, extrapolate_pose_based_on_velocity,\
                            global_vel_to_ref, extract_camera_labels, extract_lidar_labels, extract_projected_labels 
from src.common import save_pkl, load_pkl, compute_iou, compute_optimal_assignments, get_2d_bbox_corners, points_in_bboxes
from PIL import Image

class WaymoConverter(DataConverter):  
    """
    Dataset preprossing class, which preprocess waymo-open dataset to a canonical data representation as used withing the Nvidia DriveSim-AI project.
    Waymo-open data can be downloaded from https://waymo.com/intl/en_us/open/download/ in form of tfrecords files. Further details on the dataset are
    available in the original publication https://arxiv.org/abs/1912.04838 or the githbub repository https://github.com/waymo-research/waymo-open-dataset

    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com. 
    """  

    def __init__(self, args):

        # Cameras used in waymo-open
        self.camera_list = ['_FRONT', '_FRONT_LEFT', '_FRONT_RIGHT', '_SIDE_LEFT', '_SIDE_RIGHT']
       
        # Label types
        self.type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.CAMERA_2_IDTYPERIG = {1: ['00', 'pinhole', 'camera_front_50fov'],
                                   2: ['01', 'pinhole', 'camera_front_left_50fov'],
                                   3: ['02', 'pinhole', 'camera_front_right_50fov'],
                                   4: ['03', 'pinhole', 'camera_side_left_50fov'],
                                   5: ['04', 'pinhole', 'camera_side_right_50fov']}
        
        self.label_map = {0: 'unknown',
                     1: 'car',
                     2: 'pedestrian',
                     3: 'sign',
                     4: 'CYCLIST'
                     }

        super().__init__(args)

    def convert_one(self, sequence_path): 
        """
        Runs the conversion of a single sequence (approximately 20s snippet of data)
        
        Args:
            sequence_path (string): path to the raw sequence data
        """

        dataset = tf.data.TFRecordDataset(sequence_path, compression_type='')
        
        # Check that all frames in the dataset have the same sequence name (i.e. belong to the same sequence)
        for f_idx, data in enumerate(dataset):
            
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if f_idx == 0:
                sequence_name = frame.context.name
            if frame.context.name != sequence_name:
                raise SystemExit("NOT ALL FRAMES BELONG TO THE SAME SEQUENCE. ABORTING THE CONVERSION!")

        # Initialize the pose variables. Poses can be coupled to either images, lidar or default frame
        poses = []
        poses_timestamps = []
        lidar_timestamps = []
        camera_timestamps = defaultdict(list)
        annotations = {}
        annotations['3d_labels'] = defaultdict(dict)
        annotations['2d_labels'] = defaultdict(dict)
        annotations['label_corr_3d_2d'] = defaultdict(dict)
        annotations['label_corr_2d_3d'] = defaultdict(dict)

        # create all the folders
        self.create_folders(sequence_name)

        # Iterate over all the frames in a given sequence and extract the data
        for f_idx, data in enumerate(dataset):

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # Decode the lidar frame
            self.decode_lidar(frame, f_idx, lidar_timestamps, sequence_name)

            # Decode the image frames
            self.decode_images(frame, poses, poses_timestamps, camera_timestamps, f_idx, sequence_name)

            # Decode the object annotations
            self.decode_labels(frame, f_idx, annotations, sequence_name)

        # Save the sequence metadata 
        self.decode_metadata(frame, sequence_name)

        # Perform instance and semantic segmentation of all the images
        if self.sem_seg_flag:
            self.run_semantic_segmentation(sequence_name)
        
        if self.inst_seg_flag:
            self.run_instance_segmentation(sequence_name)

        # Decode the pose of the current frame and it timestamp
        self.decode_poses_timestamps(poses, poses_timestamps, camera_timestamps, lidar_timestamps, sequence_name)

        # Summarize the labels
        self.summarize_labels_across_frames(annotations, sequence_name)

        # Extract the dynamic masks
        self.extract_dynamic_masks(annotations, sequence_name)
        

    def decode_poses_timestamps(self, poses, poses_timestamps, camera_timestamps, lidar_timestamps, sequence_name):
        # Stack all the poses
        poses = np.stack(poses)
        poses_timestamps = np.stack(poses_timestamps)
        sort_idx = np.argsort(poses_timestamps)

        # All the available poses
        poses = poses[sort_idx]
        poses_timestamps = poses_timestamps[sort_idx]

        # Save the poses
        poses_save_path =  os.path.join(self.output_dir, sequence_name, 
                        self.poses_save_dir, 'poses.npz')
        np.savez(poses_save_path, ego_poses=poses, timestamps=poses_timestamps)

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

    def decode_lidar(self, frame, f_idx, lidar_timestamps, sequence_name):
        """
        Converts the raw point cloud data into 3D depth rays in space also compensating for the
        motion of the ego-car (lidar unwinding).
        
        Args:
            frame (dict): data of a single frame as defined in the waymo-open proto files
            f_idx (int): index of the frame in the sequence
            lidar_timestamps (list): timestamps of individual lidar frames, each timestamp denotes the start of the spin
            sequence_name (string): unique identifier of the sequence

        """

        # Extract the range image and corresponding poses for all rays
        range_images, camera_projections, range_image_top_pose = parse_range_image_and_camera_projection(frame)

        # Convert the range image to a ego-motion compensated 3D rays in sequence coordinate frame
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

        # Initialize the lidar save path
        lidar_save_path =  os.path.join(self.output_dir, sequence_name, 
                                        self.point_cloud_save_dir, str(f_idx).zfill(4) + '.dat')
        points = points[0]
        dist = np.linalg.norm(points[:,3:6] - points[:,:3],axis=1, keepdims=True)

        # 3D rays in space with accompanying metadata. 
        # Format; x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic flag
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
        points = np.concatenate((points[:,:6], dist, points[:,6:7], -1*np.ones_like(dist)),axis=1)
        points_flat = points.flatten()

        with open(lidar_save_path,'wb') as f:
            f.write(struct.pack('>i', points_flat.size)) # Big endian integer number of floats
            f.write(struct.pack('=%sf' % points_flat.size, *points_flat)) 

        # Extract the metadata of the lidar frame
        metadata = {}
        for lidar in frame.context.laser_calibrations:
            if lidar.name == 1: # 1 is the top lidar
                metadata['T_lidar_rig'] = np.array(lidar.extrinsic.transform).reshape(4,4)  # Lidar extrinsic parameters
                # Lidar elevation angles, can be used to simulate the lidar or recover points that did not return
                metadata['elevation_angles'] = np.array(frame.context.laser_calibrations[-1].beam_inclinations)

        # Pose of the SDC at the start of the lidar spin, can be used to transform points into a local coordinate frame
        metadata['ego_pose'] = np.reshape(np.array(frame.pose.transform), [4, 4])

        save_pkl(metadata, lidar_save_path.replace('.dat','.pkl'))

    def decode_images(self, frame, poses, poses_timestamps, camera_timestamps, f_idx, sequence_name):
        """
        Extracts the images and camera metadata for all cameras withing a single frame. Camera metadata must hold 
        the information used to compensate for rollign shutter effect and to convert RGB images to 3D RGB rays in space
        
        Args:
            frame (dict): data of a single frame as defined in the waymo-open proto files
            poses (int): poses of the SDC
            poses_timestamps (list): timestamps of each pose contained in the poses list
            camera_timestamps (list): timestamp of each image - corresponds approximately to the PP 
            f_idx (int): index of the frame in the sequence
            sequence_name (string): unique identifier of the sequence

        """

        # Sort the cameras as defaults 1) Front, 2) Front left, 3) Front right, 4) Side left, 5) Side right 
        images = sorted(frame.images, key=lambda i:i.name)
        
        for image in images:
            # Get the calibration data
            calib = frame.context.camera_calibrations[image.name - 1]

            # Get the SDC car pose 
            # TODO: Check github issues to confirm that this is correct: https://github.com/waymo-research/waymo-open-dataset/issues/464
            poses.append(np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float64), [4, 4])))
            poses_timestamps.append(image.pose_timestamp * 1e6) # Convert the poses to microseconds

            metadata = {}
            metadata['img_width'] = calib.width
            metadata['img_height'] = calib.height
            # Timestamps of the ego_pose_s and ego_pose_e defined below
            metadata['ego_pose_timestamps'] = np.array([image.camera_trigger_time, image.camera_readout_done_time])
            metadata['exposure_time'] = image.shutter

            # Rolling shutter directions expressed as an integer
            # 1 = TOP_TO_BOTTOM, 2 = LEFT_TO_RIGHT, 3 = BOTTOM_TO_TOP, 4 = RIGHT_TO_LEFT
            metadata['rolling_shutter_direction'] = calib.rolling_shutter_direction
            metadata['camera_model'] = self.CAMERA_2_IDTYPERIG[image.name][1]
            metadata['intrinsic'] = np.array(tf.constant(calib.intrinsic, dtype=tf.float64))
            metadata['T_cam_rig'] = np.array(tf.reshape(tf.constant(calib.extrinsic.transform, dtype=tf.float64), [4, 4])) # Camera to sdc

            assert metadata['rolling_shutter_direction'] in [1,2,3,4], "Weird rolling shutter direction, aborting"

            # Velocity and angular velocity of the SDC at camera pose timestamp.
            # According to proto both are expressed in the global reference frame, but code hints that the ang. velocity is in SDC frame
            # TODO: check that the angular velocity is indeed provided in the SDC frame: 
            # https://github.com/waymo-research/waymo-open-dataset/issues/462
            T_SDC_global = np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float64), [4, 4]))
            velocity_global = np.array([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z]).reshape(3,1)
            omega_vehicle = np.array([image.velocity.w_x, image.velocity.w_y, image.velocity.w_z]).reshape(3,1)
            omega_global = np.matmul(T_SDC_global[:3,:3], omega_vehicle)

            # Extrapolate the pose to the start and end timestamp of the image frame considering the (angular) velocity at the time of the acquisition
            metadata['ego_pose_s'] = extrapolate_pose_based_on_velocity(T_SDC_global,velocity_global, omega_global, metadata['ego_pose_timestamps'][0] - image.pose_timestamp)
            metadata['ego_pose_e'] = extrapolate_pose_based_on_velocity(T_SDC_global,velocity_global, omega_global, metadata['ego_pose_timestamps'][1] - image.pose_timestamp)
        
            # Save the camera pose timestamps, corresponds approximately to the timestamp of the principle point pixel
            camera_timestamps[self.CAMERA_2_IDTYPERIG[image.name][0]].append(image.pose_timestamp * 1e6)

            # Save the image and its metadata
            im = Image.fromarray(np.array(tf.image.decode_jpeg(image.image)))
    
            img_save_path =  os.path.join(self.output_dir, sequence_name, 
                             self.image_save_dir, 'image_{}'.format(self.CAMERA_2_IDTYPERIG[image.name][0]), 
                             str(f_idx).zfill(4) + '.jpeg')
            im.save(img_save_path)
    
            save_pkl(metadata, img_save_path.replace('.jpeg','.pkl'))


    def decode_labels(self, frame, f_idx, annotations, sequence_name):
        """
        Extract the per frame 2D and 3D object bounding boxes and aims to establish correspondence between them
        3D bboxes and 2D bboxes have unique IDs over the whole sequence, but the 2D-3D correspondences are missing

        
        Args:
            frame (dict): data of a single frame as defined in the waymo-open proto files
            f_idx (int): index of the frame in the sequence
            annotations (dict): Dictionary of annotations for the whole sequence
            sequence_name (string): unique identifier of the sequence

        """

        # Get the pose of the car at the start of the frame
        sdc_pose = np.reshape(np.array(frame.pose.transform), [4, 4])

         # Extract lidar labels 
        lidar_labels = extract_lidar_labels(frame)

        # Extract camera labels (annotated in 2d)
        camera_labels = extract_camera_labels(frame)

        # Extract lidar labels projected to the images
        projected_lidar_labels = extract_projected_labels(frame)

        frame_annotations = {
                    'lidar_labels': lidar_labels,
                    'camera_labels': camera_labels,
                    'projected_lidar_labels': projected_lidar_labels
        }

        anno_save_path =  os.path.join(self.output_dir, sequence_name, 
                                self.label_save_dir, str(f_idx).zfill(4) + '.pkl')

        save_pkl(frame_annotations, anno_save_path)

        # Iterate over the lidar labels and if not encountered yet, insert them into annotations otherwise ad the metadata
        # for this frame
        for label in frame.laser_labels:
            if label.id not in annotations['3d_labels']:
                # If label is not yet in the list add it and initialize the static/dynamic label 
                # all cyclists and pedestrians are assumed to be dynamic irrespective of their speed
                annotations['3d_labels'][label.id]['dynamic_flag'] = 1 if label.type in [2,4] else 0
                annotations['3d_labels'][label.id]['type'] = self.label_map[label.type]
                annotations['3d_labels'][label.id]['lidar'] = {}

            ref_velocity = global_vel_to_ref([label.metadata.speed_x, label.metadata.speed_y], sdc_pose[0:3, 0:3])

            # Insert the data and change the dynamic flag if the object is in motion
            annotations['3d_labels'][label.id]['lidar'][f_idx] = {'3D_bbox': np.array([label.box.center_x, label.box.center_y, label.box.center_z,
                                                                            label.box.length, label.box.width, label.box.height, ref_velocity[0], 
                                                                            ref_velocity[1], label.box.heading], dtype=np.float32), 
                                                                  'num_point': label.num_lidar_points_in_box, 
                                                                  'global_speed':np.array([label.metadata.speed_x, label.metadata.speed_y], dtype=np.float32), 
                                                                  'global_accel':np.array([label.metadata.accel_x, label.metadata.accel_y], dtype=np.float32)}

            # TODO: check if this user-defined threshold makes sense
            if label.type in [1,2,4] and np.max([label.metadata.speed_x, label.metadata.speed_y]) >= 0.75/3.6:
                    annotations['3d_labels'][label.id]['dynamic_flag'] = 1

        # Iterate over the 2D image labels and if not encountered yet, insert them into annotations otherwise ad the metadata
        # for this frame
        for camera in sorted(frame.camera_labels, key=lambda i:i.name):
            camera_name = 'cam_{}'.format(str(camera.name-1).zfill(2))

            for label in camera.labels:
                if label.id not in annotations['2d_labels']:
                    # If label is not yet in the list add it and initialize the static/dynamic label 
                    # all cyclists and pedestrians are assumed to be dynamic irrespective of their speed
                    annotations['2d_labels'][label.id]['dynamic_flag'] = 1 if label.type in [2,4] else 0
                    annotations['2d_labels'][label.id]['type'] = self.label_map[label.type]
                    annotations['2d_labels'][label.id][camera_name] = {}


                annotations['2d_labels'][label.id][camera_name][f_idx] = {'2D_bbox': np.array([label.box.center_x, label.box.center_y, 
                                                                            label.box.length, label.box.width], dtype=np.float32)}

        tmp_ipu = {}
        # Iterate over the labels and try to establish per-frame correspondence based on the 2D IoU (3D projected vs 2D label) 
        for camera in sorted(frame.projected_lidar_labels, key=lambda i:i.name):
            camera_name = 'cam_{}'.format(str(camera.name-1).zfill(2))
            
            for label in camera.labels:
                label_name = label.id[0:label.id.find(self.camera_list[camera.name-1])]
                proj_bbox = np.array([label.box.center_x, label.box.center_y, label.box.length, label.box.width], dtype=np.float32)

                if camera_name not in annotations['3d_labels'][label_name]:
                    annotations['3d_labels'][label_name][camera_name] = {}
                annotations['3d_labels'][label_name][camera_name][f_idx] = {'3D_bbox_proj': proj_bbox}


                for img_label in annotations['2d_labels'].keys():
                    if camera_name in annotations['2d_labels'][img_label] and \
                                        f_idx in annotations['2d_labels'][img_label][camera_name]: 
                        # Only check the IoU if both labels are of the same type
                        if self.label_map[label.type] == annotations['2d_labels'][img_label]['type']:
                            iou = compute_iou(proj_bbox, annotations['2d_labels'][img_label][camera_name][f_idx]['2D_bbox'])
                            tmp_ipu[img_label] = iou                

                # Add the label to the correspondences 
                if tmp_ipu and tmp_ipu[max(tmp_ipu, key=tmp_ipu.get)] > 0.10: # Check if there are any labels 
                    # Assign the best fitting 2D label to the 3D one
                    label_2d = max(tmp_ipu, key=tmp_ipu.get)
                    if label_name in annotations['label_corr_3d_2d'][camera_name]:
                        annotations['label_corr_3d_2d'][camera_name][label_name]['2d_name'].append(label_2d)
                        annotations['label_corr_3d_2d'][camera_name][label_name]['iou'].append(tmp_ipu[label_2d])
                    else:
                        annotations['label_corr_3d_2d'][camera_name][label_name] = {}
                        annotations['label_corr_3d_2d'][camera_name][label_name]['2d_name'] = [label_2d]
                        annotations['label_corr_3d_2d'][camera_name][label_name]['iou'] = [tmp_ipu[label_2d]]

                    # Add the assignment to the 2D label
                    if label_2d in annotations['label_corr_2d_3d'][camera_name]:
                        if label_name not in annotations['label_corr_2d_3d'][camera_name][label_2d]['name']:
                            annotations['label_corr_2d_3d'][camera_name][label_2d]['name'].append(label_name)
                        annotations['label_corr_2d_3d'][camera_name][label_2d]['count'] += 1
                    
                    else:
                        annotations['label_corr_2d_3d'][camera_name][label_2d] = {}
                        annotations['label_corr_2d_3d'][camera_name][label_2d]['name'] = [label_name]
                        annotations['label_corr_2d_3d'][camera_name][label_2d]['count'] = 1


    def summarize_labels_across_frames(self, annotations, sequence_name):
        """
        Performs the optimal transport global assignments of the 2D-3D labels based on the IoUs in individual frames
        #TODO: some thresholds are defined manually and should be evaluated on a larger dataset

        Args:
            annotations (dict): Dictionary of annotations for the whole sequence
            sequence_name (string): unique identifier of the sequence

        """

        optimal_assignments = compute_optimal_assignments(annotations['label_corr_2d_3d'], annotations['label_corr_3d_2d'], ['cam_00','cam_01','cam_02','cam_03','cam_04'])

        # Transfer the dynamic label from 3D car bboxes to their corresponding 2D labels 
        for cam in optimal_assignments.keys():
            for label_3d in optimal_assignments[cam].keys():
                if annotations['3d_labels'][label_3d]['dynamic_flag'] == 1:
                    corr_2d_label = optimal_assignments[cam][label_3d]
                    annotations['2d_labels'][corr_2d_label]['dynamic_flag'] = 1

        annotations['assignments'] = optimal_assignments
        
        # Save the accumulated data
        labels_save_path =  os.path.join(self.output_dir, sequence_name, 
                        self.label_save_dir, 'labels.pkl')
        save_pkl(annotations, labels_save_path)


    def extract_dynamic_masks(self, annotation, sequence_name):
        """
        Extract dynamic masks for both 2D images and 3D point cloud data, based on the object annotations
        #TODO: check the points_in_bboxes function as the number of points within each box does not correspond 100% to the metadata

        Args:
            annotations (dict): Dictionary of annotations for the whole sequence
            sequence_name (string): unique identifier of the sequence

        """

        # Extract the motion segmented images 
        img_folders = sorted(glob.glob(os.path.join(self.output_dir, sequence_name, self.image_save_dir, '*/')))

        for img_folder in img_folders:
            cam = 'cam_{}'.format(img_folder.split('_')[-1][:-1])
            frames = sorted(glob.glob(os.path.join(img_folder, '????.jpeg')))

            for frame_path in frames:
                img = np.array(Image.open(frame_path))
                f_idx = int(frame_path.split(os.sep)[-1].split('_')[0].split('.')[0])
                
                # create mask with zeros
                mask = np.zeros((img.shape), dtype=np.uint8)

                for label in annotation['2d_labels']:
                    if annotation['2d_labels'][label]['dynamic_flag'] == 1 and \
                        cam in annotation['2d_labels'][label] and int(f_idx) in annotation['2d_labels'][label][cam]:
                        
                        bbox = annotation['2d_labels'][label][cam][int(f_idx)]['2D_bbox']

                        # define points (as small diamond shape)
                        bbox_corners = get_2d_bbox_corners(bbox)
                        cv2.fillPoly(mask, np.int32([bbox_corners]), (255,255,255) )


                bool_mask = mask[:,:,0].astype(bool)
                bool_img = Image.fromarray(bool_mask)
                bool_img.save(os.path.join(img_folder, 'dynamic_mask_{}.jpeg'.format(str(f_idx).zfill(4))), bits=1,optimize=True)



        # Add the motion flag the to the lidar points
        dynamic_bbox = {}
        for label in annotation['3d_labels']: 
            if annotation['3d_labels'][label]['dynamic_flag'] == 1:
                dynamic_bbox[label]= annotation['3d_labels'][label]

        lidar_frames = sorted(glob.glob(os.path.join(self.output_dir, sequence_name, self.point_cloud_save_dir, '*.dat')))

        for lidar_frame in lidar_frames:
            f_idx = int(lidar_frame.split(os.sep)[-1].split('_')[0].split('.')[0])
            # Load the point clouds
            with open(lidar_frame,'rb') as f:
                # The first number denotes the number of points 
                d = f.read(4)
                n_pts = struct.unpack('>i', d)[0]
                lidar_pc = np.array(struct.unpack('=%sf' % n_pts, f.read())).reshape(-1,9)

            dynamic_flag = np.zeros(lidar_pc.shape[0])

            metadata = load_pkl(lidar_frame.replace('.dat','.pkl'))
            ego_pose_inv = np.linalg.inv(metadata['ego_pose'])
            local_pc = (ego_pose_inv[:3,:3] @ lidar_pc[:,3:6].transpose() + ego_pose_inv[:3,3:4]).transpose()

            # Load the 3D rays and set their dynamic label
            for _, label in dynamic_bbox.items():
                if f_idx in label['lidar']:
                    bbox = label['lidar'][f_idx]['3D_bbox']
                    bbox_idxs = points_in_bboxes(local_pc, bbox.reshape(1,-1))

                    dynamic_flag[bbox_idxs != -1] = 1

            lidar_pc[:,-1] = dynamic_flag
            points_flat = lidar_pc.flatten()
            with open(lidar_frame,'wb') as f:
                f.write(struct.pack('>i', points_flat.size))
                f.write(struct.pack('=%sf' % points_flat.size, *points_flat))

    
    def decode_metadata(self, frame, sequence_name):
        """
        Decodes the sequence metadata such as time of day or weather conditions, which can be used to mine
        useful data.

        Args:
            frame (dict): data of a single frame as defined in the waymo-open proto files
            sequence_name (string): unique identifier of the sequence

        """
        metadata = {'weather': frame.context.stats.weather, # Either sunny or rain
                    'time_of_day': frame.context.stats.time_of_day} # Day, Dawn/Dusk, or Night

        metadata_save_path =  os.path.join(self.output_dir, sequence_name, 
                             'metadata.pkl')

        save_pkl(metadata, metadata_save_path)