import click
import os
import glob
import json 
import copy
import numpy as np 

from scipy.spatial.transform import Rotation as R

@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed DSAI sequence to update', required=True)
@click.option('--cam_idx', '-c', type=int, help='Camera to be used.', default=-1)
@click.option('--frame_idx', '-f', type=int, help='Frame to be used.', default=-1)
@click.option('--n_poses', '-n', type=int, help='Number of poses.', default=120)
@click.option('--downsample', '-d', type=int, help='Downsampling factor.', default=1)
@click.option('--axis', '-a', type=click.Choice(['x','y','z']), help='Axis around which the rotation is performed', default='z')

def dsai_ngp_360_novel_views(root_dir: str, cam_idx: int, frame_idx: int, n_poses: int, downsample: int, axis: str):
    # Check that the path exists and containes json files
    assert os.path.exists(root_dir), "Given path does not exist."

    # Get all the json files
    all_configs = sorted([os.path.basename(file) for file in glob.glob(root_dir + '/*.json')])
    assert f'cam_{cam_idx}_train.json' in all_configs, "Config file for the selected camera does not exist!"

    config: dict = json.load(open(os.path.join(root_dir,f'cam_{cam_idx}_train.json')))
    
    # Copy the global metadata 
    global_metadata = {}
    for k,v in config.items():
        if k not in ['frames', 'lidar']:
            global_metadata[k] = v

    # Downsample if needed
    if downsample > 1:
        global_metadata['h'] /= downsample
        global_metadata['w'] /= downsample 
    
    # Check if the frame exists and copy its metadata
    frame_exists = False
    for frame in config['frames']:
        if int(os.path.basename(frame['file_path']).split('.')[0]) == frame_idx:
            frame_exists = True
            break

    assert frame_exists, f"Selected frame does not exist in the cam_{cam_idx}.json file."

    base_pose = frame['transform_matrix_start']
    sampled_transform = np.eye(4)
    angles = np.linspace(0, 2*np.pi, n_poses)

    frame_template = {}
    for k,v in frame.items():
        if ('transform' not in k) and ('file' not in k) and k != 'w' and k != 'h':
            frame_template[k] = v

    interpolated_frames = []
    for angle_idx, angle in enumerate(angles):
        rot = R.from_euler(seq=axis, angles=angle).as_matrix()
        tmp_pose = np.copy(base_pose)
        tmp_pose[:3,:3] = rot @ tmp_pose[:3,:3]

        tmp_frame = copy.deepcopy(frame_template)
        tmp_frame['transform_matrix'] = tmp_pose.tolist()
        tmp_frame['file_path'] = f'{str(angle_idx).zfill(6)}.jpeg'
        interpolated_frames.append(tmp_frame)
    
    # save the file
    save_dir = os.path.join(root_dir, 'novel_views')
    os.makedirs(save_dir, exist_ok = True)
    global_metadata['frames'] = interpolated_frames
    with open(os.path.join(save_dir,f"interpolated_360_c_{cam_idx}_f_{frame_idx}_n_{n_poses}_a_{axis}.json") , 'w') as outfile:    
        json.dump(global_metadata, outfile, indent=2)

if __name__ == "__main__":
    dsai_ngp_360_novel_views()