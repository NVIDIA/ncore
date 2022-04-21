from abs import ABC, abstractmethod
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

        self.load_dir = args.load_dir
        self.save_dir = args.save_dir
        self.num_proc = int(args.num_proc)

        if self.dataset == 'nvidia':        
            self.sequence_pathnames = sorted(glob.glob(os.path.join(self.load_dir, '*/')))

        elif self.dataset == 'waymo_open'
            self.sequence_pathnames = sorted(glob.glob(os.path.join(self.load_dir, '*.tfrecord')))


    def create_folders(self, sequence_name):

        seq_path = os.path.join(self.save_dir, sequence_name)

        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)

        for d in [self.label_save_dir,self.image_save_dir, self.poses_save_dir, self.point_cloud_save_dir, self.point_cloud_non_unwind_save_dir]:
            if not os.path.isdir(os.path.join(seq_path, d)):
                os.makedirs(os.path.join(seq_path, d))

        if self.export_unreturned_points:
            if not os.path.isdir(os.path.join(seq_path, self.all_points_dir)):
                os.makedirs(os.path.join(seq_path, self.all_points_dir))

        for camera in self.cameras:
            cam_id = self.CAMERA_2_IDTYPERIG[camera][0]
            if not os.path.isdir(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id)):
                os.makedirs(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id))

    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))
        print("\nfinished ...")


    def convert_one(self, seq_idx):
        sequence_name = self.sequence_pathnames[seq_idx].split(os.sep)[-2]
        
        # create all the folders
        self.create_folders(sequence_name)

        # Initialize the track aligned track record structure
        self.track_data = track_data_pb2.AlignedTrackRecords()

        # Read in the track record data from a proto file
        # This includes camera_records and lidar_records (see track_record proto for more detail)
        with open(os.path.join(self.sequence_pathnames[folder_idx], 'aligned_track_records.pb.txt'), 'r') as f:
            text_format.Parse(f.read(), self.track_data)

        # Extract all the lidar paths, timestamps and poses from the track record
        self.track_data = protobuf_to_dict(self.track_data)

        # Extract all poses
        self.extract_poses(sequence_name)

        # save lidar
        self.decode_lidar(sequence_name)

        # save images
        self.decode_image(sequence_name)

    @abstractmethod
    def extract_poses(self):
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