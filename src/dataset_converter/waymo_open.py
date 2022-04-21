from src.dataset_converter import DataConverter



class WaymoConverter(DataConverter):    
    def __init__(self, args):

        self.lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        # Label types
        self.type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.CAMERA_2_IDTYPERIG = {'camera_front_50fov':         ['00', 'pinhole', 'camera:front:50fov'],
                                   'camera_front_left_50fov':    ['01', 'pinhole', 'camera:front:left:50fov'],
                                   'camera_front_right_50fov':   ['02', 'pinhole', 'camera:front:right:50fov'],
                                   'camera_side_left_50fov':     ['03', 'pinhole', 'camera:side:left:50fov'],
                                   'camera_side_right_50fov':    ['04', 'pinhole', 'camera:side:right:50fov']}
