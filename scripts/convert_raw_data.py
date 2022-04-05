from google.protobuf import text_format
from protos import pointcloud_pb2
import numpy as np
import point_cloud_utils as pcu
from protobuf_to_dict import protobuf_to_dict
from common import euclidean_2_spherical_coords, extract_sensor_2_sdc, extract_pose, PoseInterpolator, save_pkl, \
                    parse_rig_sensors_from_file, sensor_to_rig, camera_intrinsic_parameters, \
                    compute_fw_polynomial, compute_ftheta_parameters, transform_point_cloud, compute_all_points, \
                    spherical_2_direction
                    
from utils import unwind_lidar
from google.protobuf import text_format
from protos import track_data_pb2
import argparse
import os
import struct 
import tqdm
import cv2
import glob
from multiprocessing import Pool

CAM2EXPOSURETIME = {
 'wide': 879.0,
 'fisheye': 5493.0
}

CAM2ROLLINGSHUTTERDELAY = {
 'wide': 31612.0,
 'fisheye': 32562.0
}


CAMERA_2_IDTYPERIG = {'camera_front_wide_120fov':    ['00', 'wide', 'camera:front:wide:120fov'],
                    'camera_cross_left_120fov':      ['01', 'wide', 'camera:cross:left:120fov'],
                    'camera_cross_right_120fov':     ['02', 'wide', 'camera:cross:right:120fov'],
                    'camera_rear_left_70fov':        ['03', 'wide', 'camera:rear:left:70fov'],
                    'camera_rear_right_70fov':       ['04', 'wide', 'camera:rear:right:70fov'],
                    'camera_rear_tele_30fov':        ['05', 'wide', 'camera:rear:tele:30fov'],
                    'camera_front_fisheye_200fov':   ['10', 'fisheye', 'camera:front:fisheye:200fov'],
                    'camera_left_fisheye_200fov':    ['11', 'fisheye', 'camera:left:fisheye:200fov'],
                    'camera_right_fisheye_200fov':   ['12', 'fisheye', 'camera:right:fisheye:200fov'],
                    'camera_rear_fisheye_200fov':    ['13', 'fisheye', 'camera:rear:fisheye:200fov']}

ID_2_CAMERA = {'00' : 'camera_front_wide_120fov',
                '01' : 'camera_cross_left_120fov',
                '02' : 'camera_cross_right_120fov',
                '03' : 'camera_rear_left_70fov',
                '04' : 'camera_rear_right_70fov',
                '05' : 'camera_rear_tele_30fov',
                '10' : 'camera_front_fisheye_200fov',
                '11' : 'camera_left_fisheye_200fov',
                '12' : 'camera_right_fisheye_200fov',
                '13' : 'camera_rear_fisheye_200fov'}

class NVIDIAConverter(object):

    def __init__(self, load_dir, save_dir, num_proc, export_img_rays, export_unreturned_points, delta_t):


        self.load_dir = load_dir
        self.save_dir = save_dir
        self.num_proc = int(num_proc)
        self.export_img_rays = export_img_rays
        self.export_unreturned_points = export_unreturned_points
        self.delta_t = delta_t

        self.cameras = ['camera_front_wide_120fov', 'camera_cross_left_120fov', 'camera_cross_right_120fov', 'camera_rear_left_70fov',
                        'camera_rear_right_70fov', 'camera_rear_tele_30fov','camera_front_fisheye_200fov', 'camera_left_fisheye_200fov',
                        'camera_right_fisheye_200fov', 'camera_rear_fisheye_200fov']
        
        # self.cameras = ['camera_front_wide_120fov']

        self.label_save_dir       = 'labels'
        self.image_save_dir       = 'images'
        self.point_cloud_save_dir = 'lidar'
        self.point_cloud_non_unwind_save_dir = 'lidar_non_unwind'
        if self.export_unreturned_points:
            self.all_points_dir = 'lidar_all_points'
        self.poses_save_dir = 'poses'

        self.sequence_pathnames = sorted(glob.glob(os.path.join(self.load_dir, '*/')))


    def create_folder(self, sequence_name):

        seq_path = os.path.join(self.save_dir, sequence_name)

        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)

        for d in [self.label_save_dir,self.image_save_dir, self.poses_save_dir, self.point_cloud_save_dir, self.point_cloud_non_unwind_save_dir]:
            if not os.path.isdir(os.path.join(seq_path, d)):
                os.makedirs(os.path.join(seq_path, d))

        if self.export_unreturned_points:
            if not os.path.isdir(os.path.join(seq_path, self.all_points_dir)):
                os.makedirs(os.path.join(seq_path, self.all_points_dir))

        for camera in self.cameras:
            cam_id = CAMERA_2_IDTYPERIG[camera][0]
            if not os.path.isdir(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id)):
                os.makedirs(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id))


    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))
        print("\nfinished ...")

    def convert_one(self, folder_idx):

        sequence_name = self.sequence_pathnames[folder_idx].split(os.sep)[-2]
        # create all the folders
        self.create_folder(sequence_name)

        # Initialize the track aligned track record structure
        self.track_data = track_data_pb2.AlignedTrackRecords()

        # Read in the track record data from a proto file
        # This includes camera_records and lidar_records (see track_record proto for more detail)
        with open(os.path.join(self.sequence_pathnames[folder_idx], 'aligned_track_records.pb.txt'), 'r') as f:
            text_format.Parse(f.read(), self.track_data)

        # Extract all the lidar paths, timestamps and poses from the track record
        self.track_data = protobuf_to_dict(self.track_data)

        # Extract all poses
        self.extract_poses(sequence_name)

        # save lidar
        self.decode_lidar(sequence_name)

        # save images
        self.decode_image(sequence_name)

    def extract_poses(self, sequence_name):

        poses = []
        pose_timestamps = []

        # Extract poses and timestamps, which are converted to the nvidia convention
        for frame in self.track_data['lidar_records'][0]['records']:
            if 'pose' in frame:
                pose_timestamps.append(frame['timestamp_micros'] + self.delta_t)
                poses.append(extract_pose(frame['pose']))

        for frame in self.track_data['camera_records'][0]['records']:
            if 'pose' in frame:
                pose_timestamps.append(frame['timestamp_micros'] + self.delta_t)
                poses.append(extract_pose(frame['pose']))

        poses = np.stack(poses)
        pose_timestamps = np.stack(pose_timestamps).astype(np.float64)

        sort_idx = np.argsort(pose_timestamps)

        # All the available poses
        
        all_poses = poses[sort_idx]
        self.all_pose_timestamps = pose_timestamps[sort_idx]
        self.base_pose = all_poses[0]
        self.all_poses = np.linalg.inv(self.base_pose) @ all_poses

        # Save all camera timestamps
        poses_save_path = os.path.join(self.save_dir, sequence_name, self.poses_save_dir, 'poses.npz')
        np.savez(poses_save_path, base_pose=self.base_pose, poses=self.all_poses, pose_timestamps=self.all_pose_timestamps)

    def decode_image(self, sequence_name):
        # Parse the rig calibration file 
        calibration_data = parse_rig_sensors_from_file(os.path.join(self.load_dir, sequence_name,'calibrated_rig_manual_corrected_chen_fixed.json'))

        # Filter the images based on the pose timestamps
        for camera in self.cameras:
            cam_id, cam_type, cam_id_rig = CAMERA_2_IDTYPERIG[camera]
            # Get the camera timestamps
            frame_timestamps = np.genfromtxt(os.path.join(self.load_dir, sequence_name, 'cameras/', camera + '.mp4.timestamps'), delimiter='\t', dtype=int)

            # Get the frame index of the first and last frame
            start_idx = np.where(frame_timestamps[:,1] > self.all_pose_timestamps[0] + CAM2ROLLINGSHUTTERDELAY[cam_type] + CAM2EXPOSURETIME[cam_type])[0][0]
            end_idx = np.where(frame_timestamps[:,1] >= self.all_pose_timestamps[-1])[0][0]

            frame_timestamps = frame_timestamps[start_idx:end_idx, :]

            # Extract all the images
            vidcap = cv2.VideoCapture(os.path.join(self.load_dir, sequence_name, 'cameras/', camera + '.mp4'))
            success, image = vidcap.read()
            count = 0
            save_frame = 0
            img_height, img_width,  = image.shape[0:2]
            while success:
                if frame_timestamps[0,0] <= count <= frame_timestamps[-1,0]:
                    save_path = os.path.join(self.save_dir, sequence_name, self.image_save_dir, 'image_' + cam_id, str(save_frame).zfill(4) + '.jpg')
                    cv2.imwrite(save_path, image)     # save frame as JPEG file   
                    save_frame += 1

                if count > frame_timestamps[-1,0]:
                    break
                success,image = vidcap.read()
                count += 1

            # Extract the metadata (get the relative transformation to the lidar sensor as the rig might change                
            T_cam_rig = sensor_to_rig(calibration_data[cam_id_rig])
            T_lidar_rig = sensor_to_rig(calibration_data['lidar:gt:top:p128:v4p5'])
            lidar_calib_path = os.path.join(args.root_dir, self.track_data['lidar_records'][0]['lidar_to_vehicle_transform_path'])
            T_lidar_sdc = extract_sensor_2_sdc(lidar_calib_path)

            # Recompute T_cam_rig
            T_cam_rig = T_lidar_sdc @ np.linalg.inv(T_lidar_rig) @ T_cam_rig

            intrinsic = camera_intrinsic_parameters(calibration_data[cam_id_rig])
            
            # Estimate the forward polynomial and other F-theta parameters
            fw_poly_coeff = compute_fw_polynomial(intrinsic)
            max_ray_distortion, max_angle = compute_ftheta_parameters(np.concatenate((intrinsic, fw_poly_coeff)))
            intrinsic =  np.concatenate((intrinsic, fw_poly_coeff, max_ray_distortion, max_angle))

            cam_pose_interpolator = PoseInterpolator(self.all_poses, self.all_pose_timestamps)

            for frame_idx, frame in enumerate(frame_timestamps):
                
                # Extract the ego car poses for interpolation 
                idx_s = np.where(frame[1] - CAM2ROLLINGSHUTTERDELAY[cam_type] - CAM2EXPOSURETIME[cam_type] > self.all_pose_timestamps)[0][-1]
                idx_e = np.where(frame[1] < self.all_pose_timestamps)[0][0]

                metadata = {}
                metadata['img_width'] = img_width
                metadata['img_height'] = img_height
                metadata['rolling_shutter_delay'] = CAM2ROLLINGSHUTTERDELAY[cam_type]
                metadata['exposure_time'] = CAM2EXPOSURETIME[cam_type]
                metadata['intrinsic'] = intrinsic
                metadata['T_cam_rig'] = T_cam_rig
                metadata['t_eof'] = frame[1]

                # Interpolate the start and end pose to the timestamps of the first and last row
                sofTimestamp = frame[1] - CAM2ROLLINGSHUTTERDELAY[cam_type]
                firstRowTimestamp = sofTimestamp - CAM2EXPOSURETIME[cam_type]
                lastRowTimestamp = frame[1] - CAM2EXPOSURETIME[cam_type]
                metadata['ego_pose_timestamps'] = np.array([firstRowTimestamp, lastRowTimestamp])
                metadata['ego_pose_s'] = cam_pose_interpolator.interpolate_to_timestamps(firstRowTimestamp)[0]
                metadata['ego_pose_e'] = cam_pose_interpolator.interpolate_to_timestamps(lastRowTimestamp)[0]
                metadata['camera_model'] = 'f_theta' if cam_type in ['wide', 'fisheye'] else 'pinhole'
    
                metadata_save_path = os.path.join(self.save_dir, sequence_name, self.image_save_dir, 'image_' + cam_id, str(frame_idx).zfill(4) + '.pkl')

                save_pkl(metadata, metadata_save_path)

            # Save all camera timestamps
            cam_timestamp_save_path = os.path.join(self.save_dir, sequence_name, self.image_save_dir, 'image_' + cam_id, 'timestamps.npz')
            np.savez(cam_timestamp_save_path, frame_t=frame_timestamps[:,1])


    def decode_lidar(self, sequence_name):

        lidar_calib_path = os.path.join(args.root_dir, self.track_data['lidar_records'][0]['lidar_to_vehicle_transform_path'])
        T_lidar_sdc = extract_sensor_2_sdc(lidar_calib_path)

        # Save the lidar to rig transformation
        lidar2rig_save_path = os.path.join(self.save_dir, sequence_name, self.poses_save_dir, 'T_lidar_rig.npz')
        np.savez(lidar2rig_save_path, T_lidar_rig=T_lidar_sdc)

        lidar_data_paths = [] 
        lidar_timestamps = []
        lidar_poses = []
        for frame in self.track_data['lidar_records'][0]['records']:
            if 'pose' in frame:
                lidar_timestamps.append(frame['timestamp_micros']  + self.delta_t)
                lidar_data_paths.append(frame['file_path'])
                lidar_poses.append(extract_pose(frame['pose']))

        # # Stack the lidar poses 
        lidar_poses = np.stack(lidar_poses)
        lidar_timestamps = np.stack(lidar_timestamps)

        # Initialize the pose interpolator object 
        pose_interpolator = PoseInterpolator(self.all_poses, self.all_pose_timestamps)

        lidar_end_timestmap = []

        # We remove the first and the last lidar frame such that the poses do not have to be extrapolated
        for frame_idx, frame_path in enumerate(lidar_data_paths[:-1]):
            # We skip the first frame such that we do not have to do extrapolation
            if frame_idx != 0:
                # Load the point cloud data
                data = pointcloud_pb2.PointCloud()
                with open(os.path.join(args.root_dir, frame_path), 'rb') as f:
                    data.ParseFromString(f.read())


                raw_pc = np.concatenate([np.array(data.data.points_x)[:,None],
                                        np.array(data.data.points_y)[:,None],
                                        np.array(data.data.points_z)[:,None]], axis=1)

    




                spherical_coordinates = euclidean_2_spherical_coords(raw_pc)           
                intensities = np.frombuffer(data.data.intensities, dtype=np.uint8)

                # Save the end time stamp of the lidar spin
                lidar_end_timestmap.append(data.meta_data.end_timestamp_microseconds + self.delta_t)

                column_timestamps = np.array(data.data.column_timestamps_microseconds) + self.delta_t
                column_poses = pose_interpolator.interpolate_to_timestamps(column_timestamps)
                T_lidar_globals = column_poses @ T_lidar_sdc[None,:,:]

                # Extract the points that did not return
                if self.export_unreturned_points:
                    lidar_all_points = compute_all_points(spherical_coordinates, intensities, np.array(data.data.column_indices), np.array(data.data.row_indices))       
                    pc_all_points = lidar_all_points[:,0:3] * lidar_all_points[:,3:4] # Direction times distance
                    transformed_pc_all_points = unwind_lidar(pc_all_points, T_lidar_globals.reshape(-1,4), lidar_all_points[:, 5:6].astype(int))
                    ray_origin = transformed_pc_all_points[:,:3]
                    ray_dirs = transformed_pc_all_points[:,3:6] - transformed_pc_all_points[:,0:3]
                    ray_dirs /= np.linalg.norm(ray_dirs,axis=1, keepdims=True)
                    ray_dist = lidar_all_points[:,3:4]
                    ray_dist[lidar_all_points[:,-1] == 0] = -1

                    all_points = np.concatenate([ray_origin, ray_dirs, ray_dist, lidar_all_points[:,4:5], -1*np.ones((ray_dirs.shape[0], 4))], axis=1)
                    all_points_flat = all_points.flatten()
                    lidar_all_points_save_path = os.path.join(self.save_dir, sequence_name, self.all_points_dir, str(frame_idx - 1).zfill(4) + '.dat')
                # lidar_non_unwind_save_path = os.path.join(self.save_dir, sequence_name, 
                #                                 self.point_cloud_non_unwind_save_dir, str(frame_idx).zfill(4) + '.dat')

                    with open(lidar_all_points_save_path,'wb') as f:
                        f.write(struct.pack('>i', all_points_flat.size))
                        f.write(struct.pack('=%sf' % all_points_flat.size, *all_points_flat))



                # compute the non-unwind point cloud
                pc_global_wo_ego_comp = transform_point_cloud(raw_pc, lidar_poses[frame_idx] @ T_lidar_sdc[:,:])

                # Filter out points that are more than 1 m bellow ground (there are some spurious measurements there)
                valid_idx_z = raw_pc[:,2] > -2.85
                transformed_pc = unwind_lidar(raw_pc, T_lidar_globals.reshape(-1,4), np.array(data.data.column_indices).reshape(-1,1))

                # Filter points with a distance smaller than 1.5m (points that lie on the ego car)
                dist = np.linalg.norm(transformed_pc[:,:3] - transformed_pc[:,3:6],axis=1)
                transformed_pc = np.concatenate([transformed_pc, dist[:,None], np.zeros_like(dist[:,None])], axis=1)

                # Filter points on the distance (remove points that are very far away and points that lie on the ego car)
                valid_idx_dist = np.logical_and(np.greater_equal(dist,3.5),np.less_equal(dist,100))
                valid_idx = np.logical_and(valid_idx_z, valid_idx_dist)

                # Filter out the invalid points
                transformed_pc = transformed_pc[valid_idx,:]
                intensities = intensities[valid_idx]
                pc_global_wo_ego_comp = pc_global_wo_ego_comp[valid_idx,:]
                transformed_pc_flat = transformed_pc.flatten()

                lidar_save_path = os.path.join(self.save_dir, sequence_name, self.point_cloud_save_dir, str(frame_idx - 1).zfill(4) + '.dat')
                # lidar_non_unwind_save_path = os.path.join(self.save_dir, sequence_name, 
                #                                 self.point_cloud_non_unwind_save_dir, str(frame_idx).zfill(4) + '.dat')

                with open(lidar_save_path,'wb') as f:
                    f.write(struct.pack('>i', transformed_pc_flat.size))
                    f.write(struct.pack('=%sf' % transformed_pc_flat.size, *transformed_pc_flat))

                pcu.save_triangle_mesh(lidar_save_path.replace('.dat', '.ply'), v=transformed_pc[:,3:6], vq=intensities)
                # pcu.save_triangle_mesh(lidar_non_unwind_save_path.replace('.dat', '.ply'), v=pc_global, vq=intensities)

                # pcu.save_mesh_v(lidar_save_path.replace('.dat', '_start.ply'), transformed_pc[:,:3])

        # Save all lidar timestamps
        lidar_timestamp_save_path = os.path.join(self.save_dir, sequence_name, self.point_cloud_save_dir, 'timestamps.npz')
        np.savez(lidar_timestamp_save_path, frame_t=lidar_timestamps[1:-1])


    def __len__(self):
        return len(self.sequence_pathnames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help="Path to the raw data")
    parser.add_argument("output_dir", type=str, help="Path where the extracted data will be saved")
    parser.add_argument('--dt_nvidia_deepmap', type=int, default=315964780260706, help='Time difference between the nvidia and deepmap timestamps')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    parser.add_argument('--export_img_rays', action='store_true', help='If image rays should be exported')
    parser.add_argument('--export_unreturned_points', action='store_true', help='If unreturned points should be exported')
    
    args = parser.parse_args()

    converter = NVIDIAConverter(args.root_dir, args.output_dir, args.num_proc, args.export_img_rays, args.export_unreturned_points, args.dt_nvidia_deepmap)
    converter.convert()
