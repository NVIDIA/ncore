import pickle 
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import spatial, interpolate
from scipy.optimize import curve_fit

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
    ''' Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations.
    https://en.wikipedia.org/wiki/Slerp

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
