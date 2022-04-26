import pickle 
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import spatial, interpolate
from scipy.optimize import linear_sum_assignment

def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path ,'rb')
    return pickle.load(file)

def save_pkl(obj, path ):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class PoseInterpolator:
    ''' 
    Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations. https://en.wikipedia.org/wiki/Slerp

    Args:
        poses (np.array): poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array): timestamps of the known poses [n]
        ts_target (np.array): timestamps for which the poses will be interpolated [m,1]
    Out:
        (np.array): interpolated poses in se3 representation [m,4,4]
    '''
    def __init__(self, poses, timestamps):

        self.slerp = spatial.transform.Slerp(timestamps, R.from_matrix(poses[:,:3,:3]))
        self.f_x = interpolate.interp1d(timestamps, poses[:,0,3])
        self.f_y = interpolate.interp1d(timestamps, poses[:,1,3])
        self.f_z = interpolate.interp1d(timestamps, poses[:,2,3])

        self.last_row = np.array([0,0,0,1]).reshape(1,1,-1)

    def interpolate_to_timestamps(self, ts_target):
        x_interp = self.f_x(ts_target).reshape(-1,1,1)
        y_interp = self.f_y(ts_target).reshape(-1,1,1)
        z_interp = self.f_z(ts_target).reshape(-1,1,1)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1,3,3)

        t_interp = np.concatenate([x_interp,y_interp,z_interp],axis=-2)

        return np.concatenate((np.concatenate([R_interp,t_interp],axis=-1), np.tile(self.last_row,(R_interp.shape[0],1,1))), axis=1)


def get_2d_bbox_corners(bbox):
    
    bbox_corners = np.zeros((4,2))
    
    bbox_corners[0,:] =  np.array([bbox[0] - 0.5 * bbox[2], bbox[1] - 0.5 * bbox[3]]).astype(np.int32) # TL
    bbox_corners[1,:]  = np.array([bbox[0] - 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]).astype(np.int32) # BL
    bbox_corners[2,:]  = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]).astype(np.int32) # BR
    bbox_corners[3,:]  = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] - 0.5 * bbox[3]]).astype(np.int32) # TR
    
    return bbox_corners

def compute_optimal_assignments(corr_2d_3d, corr_3d_2d, cameras):
    optimal_assignments = {}
    # Iterate over the cameras 
    for cam in cameras:
        optimal_assignments[cam] = {}
        # Build the cost matrix
        C = np.ones((len(corr_3d_2d[cam].keys()), len(corr_2d_3d[cam].keys()))) * 200

        tmp_labels_2d = list(corr_2d_3d[cam].keys())
        tmp_labels_3d = list(corr_3d_2d[cam].keys())

        for idx_3d, label_3d in enumerate(tmp_labels_3d):
            n_observations = len(corr_3d_2d[cam][label_3d]['2d_name'])
            all_2d_labels = list(set(corr_3d_2d[cam][label_3d]['2d_name']))

            if len(all_2d_labels) == 1 and len(corr_2d_3d[cam][all_2d_labels[0]]['name']) == 1:
                C[idx_3d, tmp_labels_2d.index(all_2d_labels[0])] = 0.0

            else:
                # Check what the ratios of the label 
                ratios_3d = []
                ratios_2d = []
                median_IoU = []
                for label in all_2d_labels:
                    n_assignments = corr_3d_2d[cam][label_3d]['2d_name'].count(label)
                    ratios_2d.append(n_assignments / corr_2d_3d[cam][label]['count'])
                    ratios_3d.append(n_assignments / n_observations)
                    median_IoU.append(np.median(corr_3d_2d[cam][label_3d]['iou'][corr_3d_2d[cam][label_3d]['2d_name'].index(label)]))

                    C[idx_3d, tmp_labels_2d.index(label)] = 200 - (10 * n_assignments * ratios_2d[-1]  * ratios_2d[-1]  * median_IoU[-1])

        
        row_ind, col_ind = linear_sum_assignment(C)

        for i in range(len(row_ind)):
            optimal_assignments[cam][tmp_labels_3d[row_ind[i]]] = tmp_labels_2d[col_ind[i]]

    return optimal_assignments

def check_overlap(bbox_1, bbox_2):

    overlap_x = np.abs(bbox_1[0] - bbox_2[0]) <= 0.5 * (bbox_1[2] + bbox_2[2])
    overlap_y = np.abs(bbox_1[1] - bbox_2[1]) <= 0.5 * (bbox_1[3] + bbox_2[3])

    return overlap_x & overlap_y


def computer_intersection_area(bbox_1, bbox_2):

    left = np.max([bbox_1[0] - bbox_1[2] * 0.5, bbox_2[0] - bbox_2[2] * 0.5])
    right = np.min([bbox_1[0] + bbox_1[2] * 0.5, bbox_2[0] + bbox_2[2] * 0.5])
    top = np.max([bbox_1[1] - bbox_1[3] * 0.5, bbox_2[1] - bbox_2[3] * 0.5])
    bottom = np.min([bbox_1[1] + bbox_1[3] * 0.5, bbox_2[1] + bbox_2[3] * 0.5])

    return  (left - right) * (top - bottom)

def compute_iou(bbox_1, bbox_2):

    # If bounding boxes do not overlap return 0.0
    if not check_overlap(bbox_1, bbox_2):
        return 0.0

    b1_area = bbox_1[2] * bbox_1[3] # Width times height
    b2_area = bbox_2[2] * bbox_2[3] # Width times height

    # Filter if 2D bbox is bigger than 3D (3D should always be bigger as it is projected, and 2d is not amodal)
    if b2_area > b1_area:
        return 0.0

    intersection_area = computer_intersection_area(bbox_1, bbox_2)
    union_area = b1_area + b2_area - intersection_area

    iou = intersection_area / union_area

    if np.max([np.min([iou, 1.0]), 0.0]) == 1.0:
        print('wrong')

    return np.max([np.min([iou, 1.0]), 0.0])


def points_in_bboxes(points, bboxes):
    """Checks whether a point is in any of the bboxes
        Args:
            points: [N, 3] tensor. Inner dims are: [x, y, z].
            boxes: [7] list of bboxes
        Returns:
            bbox_idx: [N,] idx of a bbox for each point (-1 denotes background points)
    """

    bbox_idx = -np.ones(points.shape[0])

    for crnt_ind, bbox in enumerate(bboxes):
        temp_bbox_ind = is_within_3d_bbox(points, bbox)
        bbox_idx[temp_bbox_ind] = crnt_ind

    return bbox_idx


def is_within_3d_bbox(points, box, normals=None, return_points_in_bbox_frame=False):
    """Checks whether a point is in a 3d box given a set of points and a box.
        Args:
            point: [N, 3] tensor. Inner dims are: [x, y, z].
            box: [7] tensor. Inner dims are: [center_x, center_y, center_z, length,
            width, height, heading].
            name: tf name scope.
        Returns:
            point_in_box; [N,] boolean array.
    """

    center = box[0:3]
    dim = box[3:6]
    heading = box[-1]

    # Get the rotation matrix from the heading angle
    rotation = R.from_rotvec(heading*np.array([0,0,1])).as_matrix()

    # [4, 4]
    transform = rot_trans_to_se3(rotation, center)
    # [4, 4]
    transform = np.linalg.inv(transform)
    # [3, 3]
    rotation = transform[0:3, 0:3]
    # [3]
    translation = transform[0:3, 3]

    # [M, 3]
    points_in_box_frames = np.matmul(rotation, points.transpose()).transpose() + translation

    # [M, 3]
    point_in_box = np.logical_and(
        np.logical_and(points_in_box_frames <= dim * 0.5,
                       points_in_box_frames >= -dim * 0.5),
        np.all(np.not_equal(dim, 0), axis=-1, keepdims=True))

    # [N, M]
    point_in_box = np.prod(point_in_box, axis=-1).astype(bool)

    if not return_points_in_bbox_frame:
        return point_in_box
    else:
        if normals is not None:
            T_normals = np.linalg.inv(transform).transpose()

            normals_in_bbox_frame = np.matmul(T_normals[0:3, 0:3], normals[point_in_box,:].transpose()).transpose() + T_normals[0:3, 3]

            
            return points_in_box_frames[point_in_box,:], normals_in_bbox_frame/np.linalg.norm(normals_in_bbox_frame,axis=1,keepdims=True)
        else:
            return points_in_box_frames[point_in_box,:]

def rot_trans_to_se3(rot, trans):

    if rot.ndim > 2:
        T = np.eye(4)
        T = np.tile(T,(rot.shape[0], 1, 1))
        T[:, 0:3, 0:3] = rot
        T[:, 0:3, 3] = trans.reshape(-1,3,)

    else:
        T = np.eye(4)
        T[0:3, 0:3] = rot
        T[0:3, 3] = trans.reshape(3,)

    return T