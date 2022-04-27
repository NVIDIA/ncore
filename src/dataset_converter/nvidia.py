import os
from src.dataset_converter import DataConverter
from google.protobuf import text_format
import numpy as np
from protos import track_data_pb2, pointcloud_pb2
from protobuf_to_dict import protobuf_to_dict
from src.nvidia_utils import extract_pose, extract_sensor_2_sdc
from src.common import PoseInterpolator
from lib import unwind_lidar
import glob
import struct
from pyarrow.parquet import ParquetDataset
from collections import defaultdict
from src.common import save_pkl, load_pkl, points_in_bboxes

class NvidiaConverter(DataConverter):
    def __init__(self, args):
        self.CAM2EXPOSURETIME = {'wide': 879.0, 'fisheye': 5493.0}

        self.CAM2ROLLINGSHUTTERDELAY = {'wide': 31612.0, 'fisheye': 32562.0}


        self.CAMERA_2_IDTYPERIG = {'camera_front_wide_120fov':    ['00', 'wide', 'camera:front:wide:120fov'],
                                'camera_cross_left_120fov':       ['01', 'wide', 'camera:cross:left:120fov'],
                                'camera_cross_right_120fov':      ['02', 'wide', 'camera:cross:right:120fov'],
                                'camera_rear_left_70fov':         ['03', 'wide', 'camera:rear:left:70fov'],
                                'camera_rear_right_70fov':        ['04', 'wide', 'camera:rear:right:70fov'],
                                'camera_rear_tele_30fov':         ['05', 'wide', 'camera:rear:tele:30fov'],
                                'camera_front_fisheye_200fov':    ['10', 'fisheye', 'camera:front:fisheye:200fov'],
                                'camera_left_fisheye_200fov':     ['11', 'fisheye', 'camera:left:fisheye:200fov'],
                                'camera_right_fisheye_200fov':    ['12', 'fisheye', 'camera:right:fisheye:200fov'],
                                'camera_rear_fisheye_200fov':     ['13', 'fisheye', 'camera:rear:fisheye:200fov']}

        self.ID_2_CAMERA = {'00' : 'camera_front_wide_120fov',
                            '01' : 'camera_cross_left_120fov',
                            '02' : 'camera_cross_right_120fov',
                            '03' : 'camera_rear_left_70fov',
                            '04' : 'camera_rear_right_70fov',
                            '05' : 'camera_rear_tele_30fov',
                            '10' : 'camera_front_fisheye_200fov',
                            '11' : 'camera_left_fisheye_200fov',
                            '12' : 'camera_right_fisheye_200fov',
                            '13' : 'camera_rear_fisheye_200fov'}

        self.label_map = {'unknown': 0,
                'automobile' : 1,
                'pedestrian' : 2,
                'sign' : 3,
                'CYCLIST' : 4,
                'heavy_truck': 5,
                'bus': 6,
                'other_vehicle': 7,
                'motorcycle': 8,
                'motorcycle_with_rider': 9,
                }


        super().__init__(args)

        
    def convert_one(self, sequence_path): 
        """
        Runs the conversion of a single sequence (approximately 20s snippet of data)
        
        Args:
            sequence_path (string): path to the raw sequence data
        """

        self.sequence_name = sequence_path.split(os.sep)[-2]
        
        # create all the folders
        self.create_folders(self.sequence_name)

        # Initialize the pose variables. Poses can be coupled to either images or lidar frames
        self.poses = []
        self.poses_timestamps = []
        self.lidar_timestamps = []
        self.lidar_data_paths = []
        annotations = {}
        annotations['3d_labels'] = defaultdict(dict)
        frame_annotations = defaultdict(dict)

        sequence_tracks = sorted(glob.glob(os.path.join(sequence_path,'tracks','*/')))
        for track_idx, track in enumerate(sequence_tracks):
            # Initialize the track aligned track record structure
            self.track_data = track_data_pb2.AlignedTrackRecords()

            # Read in the track record data from a proto file
            # This includes camera_records and lidar_records (see track_record proto for more detail)
            with open(os.path.join(track, 'aligned_track_records.pb.txt'), 'r') as f:
                text_format.Parse(f.read(), self.track_data)

            # Extract all the lidar paths, timestamps and poses from the track record
            self.track_data = protobuf_to_dict(self.track_data)

            final_track = (track_idx == (len(sequence_tracks) -1))
            self.decode_poses_timestamps(final_track) 

        # self.decode_labels(sequence_path, annotations, frame_annotations)

        self.decode_lidar(sequence_path)



        

    def decode_poses_timestamps(self, final_track=False):
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

        # If this is the final track, sort the poses and export them 
        if final_track:
            self.poses = np.stack(self.poses)
            self.poses_timestamps = np.stack(self.poses_timestamps).astype(np.float64)
            sort_idx = np.argsort(self.poses_timestamps)

            # All the available poses
            self.poses = self.poses[sort_idx]
            self.poses_timestamps = self.poses_timestamps[sort_idx]
            self.base_pose = self.poses[0]
            self.poses = np.linalg.inv(self.base_pose) @ self.poses

            # Save the poses
            poses_save_path = os.path.join(self.output_dir, self.sequence_name, 
                            self.poses_save_dir, 'poses.npz')
            np.savez(poses_save_path, base_pose=self.base_pose, ego_poses=self.poses, timestamps=self.poses_timestamps)

    def decode_images(self):
        pass
    def decode_labels(self, sequence_path, annotations, frame_annotations):
        
        # Read the pandas file 
        dataset = ParquetDataset(os.path.join(sequence_path, 'labels', 'autolabels.parquet'))
        table = dataset.read()
        label_data = table.to_pandas()
        label_data = label_data.reset_index()  # make sure indexes pair with number of rows

        for _, row in label_data.iterrows():
            if row['label_name'] in ['automobile', 'heavy_truck', 'bus', 'other_vehicle', 'motorcycle', 'motorcycle_with_rider']:
                
                track_id = row['trackline_id']
                label_timestamp = row['detection_timestamp']
                
                cuboid = np.array([row['centroid_x'], row['centroid_y'], row['centroid_z'], row['dim_x'],
                                   row['dim_y'], row['dim_z'], row['velocity_x'], row['velocity_y'],
                                   row['rot_x'], row['rot_y'], row['rot_z']], dtype=np.float32)
                
                if label_timestamp not in frame_annotations:
                    frame_annotations[label_timestamp]['lidar_labels'] = []

                frame_annotations[label_timestamp]['lidar_labels'].append({'id': len(frame_annotations[label_timestamp]['lidar_labels']),
                                                                            'name': track_id,
                                                                            'label': self.label_map[row['label_name']],
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
                    annotations['3d_labels'][track_id]['dynamic_flag'] = 1 if self.label_map[row['label_name']] in [2,4,8,9] else 0 
                    annotations['3d_labels'][track_id]['type'] = self.label_map[row['label_name']]
                    annotations['3d_labels'][track_id]['lidar'] = {}


                annotations['3d_labels'][track_id]['lidar'][label_timestamp] = {'3D_bbox': cuboid, 
                                                                    'num_point': -1,
                                                                    'global_speed': -1, 
                                                                    'global_accel': -1,}
            
                # TODO: check if this user-defined threshold makes sense
                if self.label_map[row['label_name']] not in [0,3] and np.max([row['velocity_x'], row['velocity_y']]) >= 0.75/3.6:
                        annotations['3d_labels'][track_id]['dynamic_flag'] = 1

        # Save the accumulated data
        labels_save_path =  os.path.join(self.output_dir, self.sequence_name, 
                        self.label_save_dir, 'labels.pkl')
        save_pkl(annotations, labels_save_path)

        # Save the accumulated data
        frame_labels_save_path =  os.path.join(self.output_dir, self.sequence_name, 
                        self.label_save_dir, 'frame_labels.pkl')
        save_pkl(frame_annotations, frame_labels_save_path)

    def decode_lidar(self, sequence_path):

        annotations  = load_pkl(os.path.join(self.output_dir, self.sequence_name, 
                        self.label_save_dir, 'labels.pkl'))
                        
        frame_annotations  = load_pkl(os.path.join(self.output_dir, self.sequence_name, 
                        self.label_save_dir, 'frame_labels.pkl'))

        fa_timestamps = np.array(sorted(list(frame_annotations.keys())))                        

        lidar_calib_path = os.path.join(sequence_path, 'to_vehicle_transform_lidar00.pb.txt')
        T_lidar_rig = extract_sensor_2_sdc(lidar_calib_path)

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

                # np.savetxt('frame_{}.txt'.format(frame_idx), np.concatenate((raw_pc, intensities.reshape(-1,1)),axis=1))
                # frame_idx += 1
                # Save the end time stamp of the lidar spin
                lidar_end_timestmap.append(data.meta_data.end_timestamp_microseconds)

                # Find the closest frame in the annotations 
                time_diff = np.abs(fa_timestamps - data.meta_data.end_timestamp_microseconds)
                new_frame_idx = np.argmin(time_diff)

                if time_diff[new_frame_idx] > 10000:
                    print("no corresponding frame found")
                    continue


                #TODO: Talk with deepmap about removing this delta t here that needs to be hardcoded
                column_timestamps = np.array(data.data.column_timestamps_microseconds) - 1319179530439720
                column_poses = pose_interpolator.interpolate_to_timestamps(column_timestamps)
                T_lidar_globals = column_poses @ T_lidar_rig[None,:,:]

                # Filter out points that are more than 1 m bellow ground (there are some spurious measurements there)
                valid_idx_z = raw_pc[:,2] > -2.85
                transformed_pc = unwind_lidar(raw_pc, T_lidar_globals.reshape(-1,4), np.array(data.data.column_indices).reshape(-1,1))

                # Filter points with a distance smaller than 1.5m (points that lie on the ego car)
                dist = np.linalg.norm(transformed_pc[:,:3] - transformed_pc[:,3:6],axis=1)
                transformed_pc = np.concatenate([transformed_pc, dist[:,None], intensities[:, None], -1*np.ones_like(dist[:,None])], axis=1)

                # Filter points on the distance (remove points that are very far away and points that lie on the ego car)
                valid_idx_dist = np.logical_and(np.greater_equal(dist,3.5),np.less_equal(dist,100))
                valid_idx = np.logical_and(valid_idx_z, valid_idx_dist)

                # 3D rays in space with accompanying metadata. 
                # Format; x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic flag
                # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
                raw_pc = raw_pc[valid_idx,:]
                transformed_pc = transformed_pc[valid_idx,:]
                
                # Use the bounding boxes to remove dynamic objects 
                
                for label in frame_annotations[fa_timestamps[new_frame_idx]]['lidar_labels']:
                    label_id = label['name']
                    dynamic_state = annotations['3d_labels'][label_id]['dynamic_flag']
                    bbox = label['3D_bbox']
                    bbox_idxs = points_in_bboxes(raw_pc, bbox.reshape(1,-1))

                    dynamic_flag[bbox_idxs != -1] = 1


                transformed_pc_flat = transformed_pc.flatten()
                lidar_save_path = os.path.join(self.output_dir, self.sequence_name, self.point_cloud_save_dir, str(frame_idx).zfill(4) + '.dat')

                with open(lidar_save_path,'wb') as f:
                    f.write(struct.pack('>i', transformed_pc_flat.size))
                    f.write(struct.pack('=%sf' % transformed_pc_flat.size, *transformed_pc_flat))

                pcu.save_triangle_mesh(lidar_save_path.replace('.dat', '.ply'), v=transformed_pc[:,3:6], vq=intensities)
                # pcu.save_triangle_mesh(lidar_non_unwind_save_path.replace('.dat', '.ply'), v=pc_global, vq=intensities)

                # pcu.save_mesh_v(lidar_save_path.replace('.dat', '_start.ply'), transformed_pc[:,:3])

                frame_idx += 1
            except:
                print("frame was not processed")
        # # Save all lidar timestamps
        lidar_timestamp_save_path = os.path.join(self.output_dir, self.sequence_name, self.point_cloud_save_dir, 'timestamps.npz')
        np.savez(lidar_timestamp_save_path, frame_t=lidar_timestamps)