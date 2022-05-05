import cv2
import os
from tqdm import tqdm
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
import sys
#sys.path.append('/home/frshen/workspace/multiview/ganverse_3d/') # Jun's repo
from copy import deepcopy
import glob


def run_instance_segmentation(img_folder):
    cfg = get_cfg()
    best_model_config = 'dependencies/instance_segmentation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
    use_point_rend = True
    if use_point_rend:
        # Download pretrained model here: https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl
        # clone the github repo to get the config file
        best_model_config = 'dependencies/instance_segmentation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
        add_pointrend_config(cfg)
        cfg.merge_from_file(best_model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = 'dependencies/instance_segmentation/pretrained_models/model_final_ba17b9.pkl'

    else:
        cfg.merge_from_file(model_zoo.get_config_file(best_model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(best_model_config)
        
    predictor = DefaultPredictor(cfg)


    all_files = sorted(glob.glob(img_folder + '????.jpg'))


    for img_name in tqdm(all_files):
        frame_num = img_name.split(os.sep)[-1].split('.')[0]
        input_image = cv2.imread(img_name)
        output = predictor(input_image)
        car_bbox = output['instances'].pred_boxes.tensor[output['instances'].pred_classes==2].cpu().numpy()
        car_mask = output['instances'].pred_masks[output['instances'].pred_classes==2].cpu().numpy()

        np.savez_compressed(os.path.join(img_folder, 'inst_seg_{}.npz'.format(frame_num)), car_bbox = car_bbox, car_mask = car_mask)