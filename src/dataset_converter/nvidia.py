from src.dataset_converter import DataConverter



class NvidiaConverter(DataConverter):
    def __init__(self, args):
        self.CAM2EXPOSURETIME = {'wide': 879.0, 'fisheye': 5493.0}

        self.CAM2ROLLINGSHUTTERDELAY = {'wide': 31612.0, 'fisheye': 32562.0}


        self.CAMERA_2_IDTYPERIG = {'camera_front_wide_120fov':    ['00', 'wide', 'camera:front:wide:120fov'],
                            'camera_cross_left_120fov':      ['01', 'wide', 'camera:cross:left:120fov'],
                            'camera_cross_right_120fov':     ['02', 'wide', 'camera:cross:right:120fov'],
                            'camera_rear_left_70fov':        ['03', 'wide', 'camera:rear:left:70fov'],
                            'camera_rear_right_70fov':       ['04', 'wide', 'camera:rear:right:70fov'],
                            'camera_rear_tele_30fov':        ['05', 'wide', 'camera:rear:tele:30fov'],
                            'camera_front_fisheye_200fov':   ['10', 'fisheye', 'camera:front:fisheye:200fov'],
                            'camera_left_fisheye_200fov':    ['11', 'fisheye', 'camera:left:fisheye:200fov'],
                            'camera_right_fisheye_200fov':   ['12', 'fisheye', 'camera:right:fisheye:200fov'],
                            'camera_rear_fisheye_200fov':    ['13', 'fisheye', 'camera:rear:fisheye:200fov']}

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

        super().__init__(args)

        