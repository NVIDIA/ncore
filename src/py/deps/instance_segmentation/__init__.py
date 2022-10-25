# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import cv2
import os
import h5py
from tqdm import tqdm
from pathlib import Path

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config

# TODO: Using Detectron2 should be fine, but we are using the models pretrained on COCO dataset here that can only be used for research purposes. Clarify the licensing or change the model in the future
def run_instance_segmentation(imgs: list):
    cfg = get_cfg()
    best_model_config = 'src/py/deps/instance_segmentation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
    use_point_rend = True
    if use_point_rend:
        # Download pretrained model here: https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl
        # clone the github repo to get the config file
        best_model_config = 'src/py/deps/instance_segmentation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
        add_pointrend_config(cfg)
        cfg.merge_from_file(best_model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = 'external/instance-segmentation-models/model_final_ba17b9.pkl'

    else:
        cfg.merge_from_file(model_zoo.get_config_file(best_model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(best_model_config)
        
    predictor = DefaultPredictor(cfg)

    for image_path in tqdm(imgs):
        input_img = cv2.imread(image_path)
        output = predictor(input_img)
        car_bbox = output['instances'].pred_boxes.tensor[output['instances'].pred_classes==2].cpu().numpy()
        car_mask = output['instances'].pred_masks[output['instances'].pred_classes==2].cpu().numpy()

        input_dir, img_name = os.path.split(image_path)
        img_name = img_name.split('.')[0]
        save_path = Path(os.path.join(input_dir,f"{img_name}_inst")).with_suffix('.hdf5')
        with h5py.File(str(save_path), "w") as f:
            COMPRESSION = 'lzf'
            f.create_dataset('car_bbox', data=car_bbox, compression=COMPRESSION)
            f.create_dataset('car_mask', data=car_mask, compression=COMPRESSION)

