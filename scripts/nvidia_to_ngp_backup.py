import numpy as np
import json
import math
import argparse
import os
import pickle
import copy

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da=da/np.linalg.norm(da)
    db=db/np.linalg.norm(db)
    c=np.cross(da,db)
    denom=(np.linalg.norm(c)**2)
    t=ob-oa 
    ta=np.linalg.det([t,db,c])/(denom+1e-10)
    tb=np.linalg.det([t,da,c])/(denom+1e-10)
    return (oa+ta*da+ob+tb*db)*0.5,denom
    
def average_camera_pose(poses):
    """
    Compute the average position of the camera
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    average_cam_position = poses[:,:3, 3].mean(0) # 
    pose_min, pose_max = np.min(poses[:,:3,3],axis=0), np.max(poses[:,:3,3],axis=0)
    extent_scene = np.max(pose_max - pose_min)

    return average_cam_position, extent_scene


def nvidia_2_ngp(args): 

    root_dir = args.root_dir
    output_dir = args.output_dir
    experiment = args.experiment
    start_frame, end_frame = [int(frame) for frame in args.frames]
    cameras = args.cameras
    export_lidar = args.use_lidar
    max_bound = args.max_bound # Max bound for scaling the scene

    nerf_2_waymo_rot = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    # nerf_2_waymo_rot = np.eye(3)
    camera_data = {}

    # Create the save path 
    if not os.path.exists(os.path.join(root_dir, 'configs/')):
        os.makedirs(os.path.join(root_dir, 'configs/'))
        
    for cam in cameras: 
        camera_data[cam] = {}
        # Load the starting frame metadata
        with open(os.path.join(root_dir, 'images/image_{}'.format(cam.zfill(2)),
                                     str(start_frame).zfill(6) + '.pkl'),'rb') as file:
            camera_metada = pickle.load(file)

        # Get the focal length, image width and height
        focal_length = camera_metada['intrinsic'][10]
        w, h = camera_metada['img_width'],  camera_metada['img_height']

        print(w,h,focal_length)

        # Compute the angular field of view
        angle_x= math.atan(w/(focal_length*2))*2
        print('fov ', angle_x*180/math.pi)
        camera_data[cam]['angle'] = angle_x

        all_img = [os.path.join(root_dir, 'images', 'image_{}'.format(cam.zfill(2)), str(idx).zfill(6) + '.jpeg') for idx in range(start_frame, end_frame)]
        all_metadata = [img.replace('.jpeg','.pkl') for img in all_img ]
        print(len(all_img))

        # Z is the up vector in the waymo coordinate system
        up=np.array([0,0,1])
        camera_data[cam]['up'] = up

        # Iterate over the poses 
        T_cam_rig = []
        poses_start = []
        poses_end = []
        # Iterate over all the frames and save their metadata
        for cam_pose in all_metadata:
            with open(cam_pose,'rb') as file:
                camera_metadata = pickle.load(file)

            T_cam_rig.append(camera_metadata['T_cam_rig'])
            T_cam_world = camera_metadata['ego_pose_s'] @ np.array(T_cam_rig)[0]
            # Iterate over the poses 
            R = T_cam_world[:3,:3] @ nerf_2_waymo_rot
            T_cam_world[:3,:3] = R
            poses_start.append(T_cam_world)

            T_cam_world = camera_metadata['ego_pose_e'] @ np.array(T_cam_rig)[0]
             # Iterate over the poses 
            R = T_cam_world[:3,:3] @ nerf_2_waymo_rot
            T_cam_world[:3,:3] = R
            poses_end.append(T_cam_world)
                   
        poses_start = np.stack(poses_start)
        poses_end = np.stack(poses_end)

        # Concatenate the bottom row
        camera_data[cam]['poses_start'] = poses_start
        camera_data[cam]['poses_end'] = poses_end
        camera_data[cam]['intrinsic'] =  camera_metadata['intrinsic']

        # Extract the rolling shutter parameters such y = a + b*x + c*y 
        t_duration = camera_metadata['ego_pose_timestamps'][1] - camera_metadata['ego_pose_timestamps'][0]
        # half_exposure = camera_metadata['exposure_time'] / 2
        # # half_exposure = 99961.39 / 2
        # x_0 = 0.5/h
        # y_0 = half_exposure/t_duration
        # x_1 = (h-0.5)/h
        # y_1 = (t_duration - half_exposure)/t_duration

        # c = (y_1 - y_0)/(x_1- x_0)
        # a = y_0 - c * x_0
        # b = 0
        a = b = 0
        c = 1.0
        camera_data[cam]['rolling_shutter'] = np.array([a, b, c])

        # camera_data[cam]['rolling_shutter'] = np.array([-(0.5)/(h - 1), 0, (1)/(1 - 1/h)])

    # Combine all the poses and compute the scaling factor and centroid, use the end timestamp pose as approximation
    all_poses = []
    for cam in cameras: 
        all_poses.append(camera_data[cam]['poses_start'])
    all_poses = np.concatenate(all_poses,axis=0)

    pose_avg, extent = average_camera_pose(all_poses)
    scale_factor = 1/((extent/2 + max_bound)/3.0) # so that the max far is scaled to 5
    offset = -(pose_avg * scale_factor) + np.array([0.5,0.5,0.5])
    

    if args.move_up or args.move_left or args.move_down or args.move_right:
        for cam in cameras: 
            camera_data[cam] = {}
            # Load the starting frame metadata
            with open(os.path.join(root_dir, 'images/image_{}'.format(cam.zfill(2)),
                                        str(start_frame).zfill(6) + '.pkl'),'rb') as file:
                camera_metada = pickle.load(file)

            # Get the focal length, image width and height
            focal_length = camera_metada['intrinsic'][10]
            w, h = camera_metada['img_width'],  camera_metada['img_height']

            print(w,h,focal_length)

            # Compute the angular field of view
            angle_x= math.atan(w/(focal_length*2))*2
            print('fov ', angle_x*180/math.pi)
            camera_data[cam]['angle'] = angle_x

            all_img = [os.path.join(root_dir, 'images', 'image_{}'.format(cam.zfill(2)), str(idx).zfill(6) + '.jpeg') for idx in range(start_frame, end_frame)]
            all_metadata = [img.replace('.jpeg','.pkl') for img in all_img ]
            print(len(all_img))

            # Z is the up vector in the waymo coordinate system
            up=np.array([0,0,1])
            camera_data[cam]['up'] = up

            # Iterate over the poses 
            T_cam_rig = []
            poses_start = []
            poses_end = []
            # Iterate over all the frames and save their metadata
            for cam_pose in all_metadata:
                with open(cam_pose,'rb') as file:
                    camera_metadata = pickle.load(file)

                if args.move_up:
                    camera_metadata['T_cam_rig'][2,3] += 0.5
                if args.move_down:
                    camera_metadata['T_cam_rig'][2,3] -= 0.5
                if args.move_left:
                    camera_metadata['T_cam_rig'][1,3] += 0.5
                if args.move_right:
                    camera_metadata['T_cam_rig'][1,3] -= 0.5


                T_cam_rig.append(camera_metadata['T_cam_rig'])
                T_cam_world = camera_metadata['ego_pose_s'] @ np.array(T_cam_rig)[0]
                # Iterate over the poses 
                R = T_cam_world[:3,:3] @ nerf_2_waymo_rot
                T_cam_world[:3,:3] = R
                poses_start.append(T_cam_world)

                T_cam_world = camera_metadata['ego_pose_e'] @ np.array(T_cam_rig)[0]
                # Iterate over the poses 
                R = T_cam_world[:3,:3] @ nerf_2_waymo_rot
                T_cam_world[:3,:3] = R
                poses_end.append(T_cam_world)
                    
            poses_start = np.stack(poses_start)
            poses_end = np.stack(poses_end)

            # Concatenate the bottom row
            camera_data[cam]['poses_start'] = poses_start
            camera_data[cam]['poses_end'] = poses_end
            camera_data[cam]['intrinsic'] =  camera_metadata['intrinsic']

            # Extract the rolling shutter parameters such y = a + b*x + c*y 
            t_duration = camera_metadata['ego_pose_timestamps'][1] - camera_metadata['ego_pose_timestamps'][0]
            # half_exposure = camera_metadata['exposure_time'] / 2
            # # half_exposure = 99961.39 / 2
            # x_0 = 0.5/h
            # y_0 = half_exposure/t_duration
            # x_1 = (h-0.5)/h
            # y_1 = (t_duration - half_exposure)/t_duration

            # c = (y_1 - y_0)/(x_1- x_0)
            # a = y_0 - c * x_0
            # b = 0
            a = b = 0
            c = 1.0
            camera_data[cam]['rolling_shutter'] = np.array([a, b, c])

        # camera_data[cam]['rolling_shutter'] = np.array([-(0.5)/(h - 1), 0, (1)/(1 - 1/h)])
    # Rescale and move the poses
    for cam_idx, cam in enumerate(cameras): 

        out_train={"aabb_scale":16, "n_extra_learnable_dims" : 8, "camera_angle_x":camera_data[cam]['angle'], "up":camera_data[cam]['up'].tolist(), "offset": offset.tolist(),
            "scale": scale_factor,"max_bound": max_bound, "enable_ray_loading": True, "cx": float(camera_data[cam]['intrinsic'][0]), "cy": float(camera_data[cam]['intrinsic'][1]),
            "w": float(camera_data[cam]['intrinsic'][2]) , "h": float(camera_data[cam]['intrinsic'][3]), "ftheta_p0": float(camera_data[cam]['intrinsic'][4]),
            "ftheta_p1": float(camera_data[cam]['intrinsic'][5]),"ftheta_p2": float(camera_data[cam]['intrinsic'][6]),"ftheta_p3": float(camera_data[cam]['intrinsic'][7]),
            "ftheta_p4": float(camera_data[cam]['intrinsic'][8]),"rolling_shutter":camera_data[cam]['rolling_shutter'].tolist(),
            
            "frames":[]}

        out_test = copy.deepcopy(out_train)

        all_img = [os.path.join(root_dir, 'images', 'image_{}'.format(cam.zfill(2)), str(idx).zfill(6) + '.jpeg') for idx in range(start_frame, end_frame)]
        all_img_train = [os.path.join(root_dir, 'images', 'image_{}'.format(cam.zfill(2)), str(idx).zfill(6) + '.jpeg') for idx in range(start_frame, end_frame, args.frame_step)]
        
        for i, name in enumerate(all_img_train):
            path = os.sep.join(name.split(os.sep)[-3:])
            # path = name
            frame={"file_path":os.path.join('..','..',path), 
            "transform_matrix_start": camera_data[cam]['poses_start'][args.frame_step * i].tolist(), 
            "transform_matrix_end": camera_data[cam]['poses_end'][args.frame_step * i].tolist(), 
            }


            out_train['frames'].append(frame)

        for i, name in enumerate(all_img):
            path = os.sep.join(name.split(os.sep)[-3:])
            # path = name
            frame={"file_path":os.path.join('..','..',path), 
            "transform_matrix_start": camera_data[cam]['poses_start'][i].tolist(), 
            "transform_matrix_end": camera_data[cam]['poses_end'][i].tolist(), 
            }


            out_test['frames'].append(frame)

        if cam_idx == 0 and export_lidar:
            out_train['lidar'] = []
            # all_lidar = [os.path.join(root_dir, 'configs/lidar', '{}.dat'.format(str(idx).zfill(6)))]
            camera_timestamps = pickle.load(open((os.path.join(root_dir, 'images/timestamps.pkl')),'rb'))['00']
            lidar_timestamps = np.load(os.path.join(root_dir, 'lidar/timestamps.npz'))['frame_t']
            start_timestamp = camera_timestamps[start_frame]
            end_timestamp = camera_timestamps[end_frame]
            lidar_start_idx  = np.where(lidar_timestamps > start_timestamp)[0][0] + 1
            lidar_end_idx   = np.where(lidar_timestamps < end_timestamp)[0][-1]  + 1
            all_lidar = [os.path.join(root_dir, 'lidar', '{}.dat'.format(str(idx).zfill(6))) for idx in range(lidar_start_idx, lidar_end_idx + 50)]

            for i, name in enumerate(all_lidar):
                path =  os.sep.join(name.split(os.sep)[-2:])
                # path = name
                frame={"file_path": os.path.join('..','..',path)}
                out_train['lidar'].append(frame)


        if not os.path.exists(os.path.join(output_dir,experiment)):
            os.makedirs(os.path.join(output_dir,experiment))
            
        print('writing train camera.json...')
        ending = ''
        if args.move_up:
            ending += '_up'
        if args.move_down:
            ending += '_down'
        if args.move_left:
            ending += '_left'
        if args.move_right:
            ending += '_right'

        with open(os.path.join(output_dir,experiment, f'transforms_{experiment}_cam_{cam}_train{ending}.json'), 'w') as outfile:    
            json.dump(out_train, outfile, indent=2)

        print('writing test camera.json...')
        keys_to_delete = []
        for key in out_test.keys():
            if 'ftheta' in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del out_test[key]

        with open(os.path.join(output_dir,experiment, f'transforms_{experiment}_cam_{cam}_test{ending}.json'), 'w') as outfile:    
            json.dump(out_test, outfile, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='Directory to the extracted waymo_open data', default='/home/zgojcic/code/av_processing/nvidia_data/extracted_data/44093')
    parser.add_argument('--output_dir', help='Directory to the extracted waymo_open data', default='/home/zgojcic/Downloads/nvidia_dataset_ngp')
    parser.add_argument('--experiment', help='Directory where the output json files will be saved')
    parser.add_argument("--frames", nargs="+", default=[900, 950], help='Indices of the start and end frame')
    parser.add_argument("--frame_step", type=int, default=3, help='Steps in which the frames will be used')
    parser.add_argument("--cameras", nargs="+", default=['1'], help='Indices of the cameras to be used')
    parser.add_argument("--max_bound", type=float, default=150., help='Maximum ranges of the cameras')
    parser.add_argument("--use_lidar", action='store_true', help='If set, lidar metadata will be exported and ready to use')
    parser.add_argument("--val_ratio", type=float, help='Ratio of the images used for validation', default=0.01)
    parser.add_argument("--move_up", action='store_true')
    parser.add_argument("--move_down", action='store_true')
    parser.add_argument("--move_left", action='store_true')
    parser.add_argument("--move_right", action='store_true')
    args = parser.parse_args()

    print(args.move_left)
    nvidia_2_ngp(args)