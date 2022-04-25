from abc import ABC, abstractmethod
from multiprocessing import Pool
import tqdm 
import os
import glob 

class DataConverter(ABC):
    def __init__(self, args):

        self.dataset = args.dataset
        self.label_save_dir       = 'labels'
        self.image_save_dir       = 'images'
        self.point_cloud_save_dir = 'lidar'
        self.poses_save_dir = 'poses'

        self.root_dir = args.root_dir
        self.output_dir = args.output_dir
        self.num_proc = int(args.n_proc)

        if self.dataset == 'nvidia':        
            self.sequence_pathnames = sorted(glob.glob(os.path.join(self.root_dir, '*/')))

        elif self.dataset == 'waymo_open':
            self.sequence_pathnames = sorted(glob.glob(os.path.join(self.root_dir, '*.tfrecord')))


    def create_folders(self, sequence_name):

        seq_path = os.path.join(self.output_dir, sequence_name)

        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)

        for d in [self.label_save_dir,self.image_save_dir, self.poses_save_dir, self.point_cloud_save_dir]:
            if not os.path.isdir(os.path.join(seq_path, d)):
                os.makedirs(os.path.join(seq_path, d))

        # if self.export_unreturned_points:
        #     if not os.path.isdir(os.path.join(seq_path, self.all_points_dir)):
        #         os.makedirs(os.path.join(seq_path, self.all_points_dir))

        for cam in self.CAMERA_2_IDTYPERIG.keys():
            cam_id = self.CAMERA_2_IDTYPERIG[cam][0]
            if not os.path.isdir(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id)):
                os.makedirs(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id))

    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = tqdm.tqdm(p.map(self.convert_one, self.sequence_pathnames))
        print("\nfinished ...")

    def run_semantic_segmentation(self, sequence_name):
        image_folders = glob.glob(os.path.join(self.output_dir, sequence_name, self.image_save_dir) + '*/')

        for img_folder in img_folders:
            pass



    @abstractmethod
    def convert_one(self, sequence_path):
        pass
    @abstractmethod
    def decode_poses_timestamps(self):
        pass

    @abstractmethod
    def decode_lidar(self):
        pass

    @abstractmethod
    def decode_images(self):
        pass
    
    @abstractmethod
    def decode_labels(self):
        pass