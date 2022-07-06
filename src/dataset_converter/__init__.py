# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from abc import ABC, abstractmethod
from multiprocessing import Pool
import tqdm 
import os
import glob 
from PIL import Image
import subprocess
import shutil
import logging
from dependencies.instance_segmentation.run_instance_segmentation import run_instance_segmentation
from dependencies.surface_reconstruction.run_surface_reconstruction import run_surface_reconstruction

# Initialize basic top-level logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s')
                    
class DataConverter(ABC):
    '''
    Base preprocessing class used to preprocess AV datasets in a canonical representation as used in the Nvidia DriveSim-AI project. For adding a new dataset,
    please inherit this class and implement the required functions. The output data should follow the conventions defined in 
    https://gitlab-master.nvidia.com/zgojcic/drivesim-ai/-/blob/main/docs/data.md

    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com. 
    '''  

    INDEX_DIGITS = 6 # the number of integer digits to pad counters in output filenames to

    class Config(object):
        """ Simple dictionary holding all options as key/value pairs """

        def __init__(self, kwargs):
            self.__dict__ = kwargs

        def __iadd__(self, other):
            """ Extend with more key/value options """
            for key, value in other.items():
                self.__dict__[key] = value

            return self


    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        self.label_save_dir = 'labels'
        self.image_save_dir = 'images'
        self.point_cloud_save_dir = 'lidar'
        self.poses_save_dir = 'poses'
        self.rec_save_dir = 'reconstructed_surface'

        self.root_dir = config.root_dir
        self.output_dir = config.output_dir
        self.num_proc = config.n_proc

        self.sem_seg_flag = config.semantic_seg
        self.inst_seg_flag = config.instance_seg
        self.surf_rec_flag = config.surface_rec


    def create_folders(self, sequence_name):
        ''' 
        Creates the default folder structure for a given sequence

        Args: 
            sequence_name (string): unique identifier of the sequence
        '''
        
        seq_path = os.path.join(self.output_dir, sequence_name)

        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)

        for d in [self.label_save_dir,self.image_save_dir, self.poses_save_dir, self.point_cloud_save_dir, self.rec_save_dir]:
            if not os.path.isdir(os.path.join(seq_path, d)):
                os.makedirs(os.path.join(seq_path, d))

        for cam in self.CAMERA_2_IDTYPERIG.keys():
            cam_id = self.CAMERA_2_IDTYPERIG[cam][0]
            if not os.path.isdir(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id)):
                os.makedirs(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id))

    def convert_sequence(self, sequence_pathname):
        # Perform all data-specific conversions
        for sub_sequence_name in self.convert_one(sequence_pathname):
            ## Perform all generic conversions
            
            # Perform instance and semantic segmentation of all the images (if enabled)
            if self.sem_seg_flag:
                self.run_semantic_segmentation(sub_sequence_name)
            
            if self.inst_seg_flag:
                self.run_instance_segmentation(sub_sequence_name)

            # Perform surface reconstruction (if enabled)
            if self.surf_rec_flag:
                self.run_surface_extraction(sub_sequence_name)
                
    def convert(self):
        self.logger.info("start converting ...")
        if self.num_proc > 1:
            # Perform multi-threaded conversion
            with Pool(self.num_proc) as p:
                r = tqdm.tqdm(p.map(self.convert_sequence,
                              self.sequence_pathnames))
        else:
            # Perform single-threaded conversion in main thread
            for sequence_pathname in self.sequence_pathnames:
                self.convert_sequence(sequence_pathname)
        self.logger.info("finished conversion ...")

    def run_semantic_segmentation(self, sequence_name):
        img_folders = glob.glob(os.path.join(self.output_dir, sequence_name, self.image_save_dir) + '/*/')
        
        for img_folder in img_folders:
            imgs = sorted(glob.glob(img_folder + f"{'?'*self.INDEX_DIGITS}.jpeg"))

            # Create a temporary folder 
            if not os.path.exists(os.path.join(img_folder, 'tmp_img')):
                os.makedirs(os.path.join(img_folder, 'tmp_img'))
            
            # Save the target resolutions
            img_res = []
            for file in imgs:
                img = Image.open(file)
                w,h = img.size[0], img.size[1]
                img_res.append((w,h))

                # Resize if the image is to large
                if w > 1920 or h > 1280:
                    img = img.resize((w//2,h//2), Image.ANTIALIAS)
                img.save(os.path.join(img_folder, 'tmp_img', file.split(os.sep)[-1]))

            args =  f'--dataset cityscapes --cv 0 --fp16 --bs_val 1 --eval folder ' \
                    '--eval_folder {} --n_scales 0.5,1.0,2.0 '\
                    '--snapshot dependencies/semantic-segmentation/pretrained_models/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth '\
                    '--arch ocrnet.HRNet_Mscale --result_dir {}'.format(os.path.join(img_folder, 'tmp_img'),os.path.join(img_folder, 'tmp_img','semantic_seg'))

            # Run the semantic segmentation
            cmd = 'python dependencies/semantic-segmentation/train.py ' + args
            subprocess.Popen(cmd, shell=True).wait()

            predictions = sorted(glob.glob(os.path.join(img_folder, 'tmp_img','semantic_seg','best_images', f"{'?'*self.INDEX_DIGITS}_prediction.png")))

            assert len(predictions) == len(img_res), "Number of semantic segmentation predictions is not the same as the number of input images"

            for idx, pred_img in enumerate(predictions):
                img = Image.open(pred_img)
                w,h = img.size[0], img.size[1]
                
                if w != img_res[idx][0] or h != img_res[idx][1]:
                    img = img.resize(img_res[idx], Image.ANTIALIAS)
                
                frame_num = pred_img.split(os.sep)[-1].split('_')[0]
                img.save(os.path.join(img_folder, 'sem_seg_{}.png'.format(frame_num)))

            # Delete the temporary file
            shutil.rmtree(os.path.join(img_folder, 'tmp_img'))


    def run_instance_segmentation(self, sequence_name):
        img_folders = glob.glob(os.path.join(self.output_dir, sequence_name, self.image_save_dir) + '/*/')
        
        for img_folder in img_folders:
            run_instance_segmentation(sorted(glob.glob(img_folder + f"{'?'*self.INDEX_DIGITS}.jpeg")))


    def run_surface_extraction(self,sequence_name):
        pc_folder = os.path.join(self.output_dir, sequence_name, self.point_cloud_save_dir)
        
        run_surface_reconstruction(pc_folder, os.path.join(self.output_dir, sequence_name, self.rec_save_dir))

    @abstractmethod
    def convert_one(self, sequence_path):
        """
        Runs dataset-specific conversion

        Args:
            sequence_path (string): path to dataset-specific raw sequence data
        
        Return:
            sub_sequence_names List[string]: names of all processed sub-sequences
        """
        pass


class BaseNvidiaDataConverter(DataConverter):
    """
    Base class for all Nvidia-specific data converters, maintaining common definitions and logic
    """

    ## Constants defined for *Hyperion8* sensor-set

    # TODO: the value for the 70FoV wide camera seems to be different, we need to clarify
    CAM2EXPOSURETIME = {'wide': 1641.58, 'fisheye': 10987.00}

    CAM2ROLLINGSHUTTERDELAY = {'wide': 31611.55, 'fisheye': 32561.63}

    CAMERA_2_IDTYPERIG = {
        'camera_front_wide_120fov': ['00', 'wide', 'camera:front:wide:120fov'],
        'camera_cross_left_120fov': ['01', 'wide', 'camera:cross:left:120fov'],
        'camera_cross_right_120fov': ['02', 'wide', 'camera:cross:right:120fov'],
        'camera_rear_left_70fov': ['03', 'wide', 'camera:rear:left:70fov'],
        'camera_rear_right_70fov': ['04', 'wide', 'camera:rear:right:70fov'],
        'camera_rear_tele_30fov': ['05', 'wide', 'camera:rear:tele:30fov'],
        'camera_front_fisheye_200fov': ['10', 'fisheye', 'camera:front:fisheye:200fov'],
        'camera_left_fisheye_200fov': ['11', 'fisheye', 'camera:left:fisheye:200fov'],
        'camera_right_fisheye_200fov': ['12', 'fisheye', 'camera:right:fisheye:200fov'],
        'camera_rear_fisheye_200fov': ['13', 'fisheye', 'camera:rear:fisheye:200fov']
    }

    ID_TO_CAMERA = {'00': 'camera_front_wide_120fov',
                    '01': 'camera_cross_left_120fov',
                    '02': 'camera_cross_right_120fov',
                    '03': 'camera_rear_left_70fov',
                    '04': 'camera_rear_right_70fov',
                    '05': 'camera_rear_tele_30fov',
                    '10': 'camera_front_fisheye_200fov',
                    '11': 'camera_left_fisheye_200fov',
                    '12': 'camera_right_fisheye_200fov',
                    '13': 'camera_rear_fisheye_200fov'
                    }

    # Reference lidar sensor name
    LIDAR_SENSORNAME = 'lidar:gt:top:p128:v4p5'

    # Minimum / maximum distances (in meters) for point cloud measurements (to filter out invalid points, points on the ego-car),
    # as well as minimum height (there might be some spurious measurements bellow ground)
    LIDAR_FILTER_MIN_DISTANCE = 3.5
    LIDAR_FILTER_MAX_DISTANCE = 100.0
    LIDAR_FILTER_MIN_RIG_HEIGHT = -0.5
