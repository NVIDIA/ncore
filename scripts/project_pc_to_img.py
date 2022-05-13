import pickle
import os 
import torch
import point_cloud_utils as pcu
import numpy as np
from PIL import Image
from src.nvidia_utils import transform_point_cloud, PoseInterpolator, world_points_2_pixel, project_camera_rays_2_img
from matplotlib import pyplot as plt
# from utils import cameraRay2Pixel, rollingShutterProjection
from src.common import load_pc_dat
import time

def read_image_colors(image_path):
    colors = np.array(Image.open(image_path).convert('RGB'))
    return colors

def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(camera_image)
  plt.grid("off")

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
  plt.axis('off')
  plt.grid(b=None)


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


cam_frame_num = 5540
root_dir = '/scratch/data/nvidia/processed_data/ada35b96-3576-11ec-9508-00044bf65d8d/45574'
camera = 'camera_front_wide_120fov'

# Load image
img = read_image_colors(os.path.join(root_dir, 'images/image_{}'.format(CAMERA_2_IDTYPERIG[camera][0]),
                       str(cam_frame_num).zfill(4) + '.jpeg'))

# Load metadata
with open(os.path.join(root_dir, 'images/image_{}'.format(CAMERA_2_IDTYPERIG[camera][0]),
                       str(cam_frame_num).zfill(4) + '.pkl'), 'rb') as f:
    metadata = pickle.load(f)


# Load point cloud 
# Find the closest lidar frame based on the timestamp
t_eof = metadata['ego_pose_timestamps'][1] - metadata['exposure_time']/2

lidar_timestamps = np.load(os.path.join(root_dir, 'lidar/timestamps.npz'))['frame_t']
lidar_frame_idx = np.argmin(np.abs(lidar_timestamps - t_eof))


pc = load_pc_dat(os.path.join(root_dir, 'lidar', f'{str(lidar_frame_idx).zfill(4)}.dat'))
pc = pc[:,3:6]

# Project the points without considering rolling shutter
pose_timestamps = metadata['ego_pose_timestamps']
poses = np.stack((metadata['ego_pose_s'], metadata['ego_pose_e']))
pose_interpolator = PoseInterpolator(poses, pose_timestamps)

cam_pose_global = pose_interpolator.interpolate_to_timestamps(t_eof)

single_pose = np.linalg.inv(metadata['T_cam_rig']) @ np.linalg.inv(cam_pose_global[0]) 
pc_cam = transform_point_cloud(pc, single_pose)

pixel_coords, valid_idx = project_camera_rays_2_img(pc_cam, metadata)
pixel_coords = pixel_coords[valid_idx]
pc_cam = pc_cam[valid_idx]

# Filter out points behind the camera
frontIdx = np.where(pixel_coords[:,2] > 0.0)[0]
pixel_coords = pixel_coords[frontIdx,:2]
pc_cam = pc_cam[frontIdx]

# Compute the distance to the points in the camera coordinate system
dist = np.linalg.norm(pc_cam,axis=1,keepdims=True)

# Project the points by considering rolling shutter
start_time = time.time()
pixel_coords_rs, trans_matrices_rs, valid_idx_rs = world_points_2_pixel(pc, metadata)
print("C++ imp requires: {} s for the rolling shutter projection of {} points".format(time.time() - start_time, pc.shape[0]))

# Compute the distance to the points in the camera coordinate system
transformed_points = (trans_matrices_rs[:,:3,:3] @ pc[valid_idx_rs,:,None] + trans_matrices_rs[:,:3,3:4]).squeeze(-1)
dist_rs = np.linalg.norm(transformed_points,axis=1,keepdims=True)

plot_points_on_image(np.concatenate((pixel_coords, dist),axis=1), img, rgba, point_size=8.0)
plot_points_on_image(np.concatenate((pixel_coords_rs[:,:2], dist_rs),axis=1), img, rgba, point_size=8.0)

print('done')