from abc import ABC, abstractmethod
from multiprocessing import Pool
import tqdm 
import os
import glob 
from PIL import Image
import subprocess
import shutil
from dependencies.instance_segmentation.run_instance_segmentation import run_instance_segmentation

class DataConverter(ABC):
    '''
    Base preprossing class used to preprocess AV datasets in a canonical representation as used in the Nvidia DriveSim-AI project. For adding a new dataset,
    please inherit this class and implement the required functions. The output data should follow the conventions defined in 
    https://gitlab-master.nvidia.com/zgojcic/drivesim-ai/-/blob/main/docs/data.md

    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com. 
    '''  

    def __init__(self, args):
        self.dataset = args.dataset
        self.label_save_dir       = 'labels'
        self.image_save_dir       = 'images'
        self.point_cloud_save_dir = 'lidar'
        self.poses_save_dir = 'poses'

        self.root_dir = args.root_dir
        self.output_dir = args.output_dir
        self.num_proc = int(args.n_proc)

        self.sem_seg_flag = args.semantic_seg
        self.inst_seg_flag = args.instance_seg

        if self.dataset == 'nvidia':        
            self.sequence_pathnames = sorted(glob.glob(os.path.join(self.root_dir, '*/')))

        elif self.dataset == 'waymo_open':
            self.sequence_pathnames = sorted(glob.glob(os.path.join(self.root_dir, '*.tfrecord')))


    def create_folders(self, sequence_name):
        ''' 
        Creates the default folder structure for a given sequence

        Args: 
            sequence_name (string): unique identifier of the sequence
        '''
        
        seq_path = os.path.join(self.output_dir, sequence_name)

        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)

        for d in [self.label_save_dir,self.image_save_dir, self.poses_save_dir, self.point_cloud_save_dir]:
            if not os.path.isdir(os.path.join(seq_path, d)):
                os.makedirs(os.path.join(seq_path, d))

        for cam in self.CAMERA_2_IDTYPERIG.keys():
            cam_id = self.CAMERA_2_IDTYPERIG[cam][0]
            if not os.path.isdir(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id)):
                os.makedirs(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id))

    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
             r = tqdm.tqdm(p.map(self.convert_one, self.sequence_pathnames[7:8]))
        print("\nfinished ...")

    def run_semantic_segmentation(self, sequence_name):
        img_folders = glob.glob(os.path.join(self.output_dir, sequence_name, self.image_save_dir) + '/*/')
        
        for img_folder in img_folders:
            imgs = sorted(glob.glob(img_folder + '????.jpeg'))

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

            predictions = sorted(glob.glob(os.path.join(img_folder, 'tmp_img','semantic_seg','best_images', '????_prediction.png')))

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
            run_instance_segmentation(img_folder)



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