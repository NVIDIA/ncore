# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from typing import Tuple
from pathlib import Path

import h5py
import tqdm
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config

from dsai_internal.data.types import EncodedImageHandle
from dsai_internal.data.util import padded_index_string


# TODO: Using Detectron2 should be fine, but we are using the models pretrained on COCO dataset here that can only be used for research purposes. Clarify the licensing or change the model in the future
def run_instance_segmentation(image_handles: list[Tuple[int, EncodedImageHandle]],
                              output_dir: Path,
                              index_digits: int,
                              use_pointrend=True):
    cfg = get_cfg()
    if use_pointrend:
        best_model_config = 'src/dsai_internal/deps/instance_segmentation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
        add_pointrend_config(cfg)
        cfg.merge_from_file(best_model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = 'external/instance-segmentation-models/model_final_ba17b9.pkl'
    else:
        best_model_config = 'src/dsai_internal/deps/instance_segmentation/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
        cfg.merge_from_file(model_zoo.get_config_file(best_model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(best_model_config)

    predictor = DefaultPredictor(cfg)

    for image_handle in tqdm.tqdm(image_handles):
        img_id = image_handle[0]
        input_img_rgb = np.asarray(image_handle[1].get_data().get_decoded_image())
        input_img_bgr = input_img_rgb[..., ::-1]  # invert last dimension from RGB -> BGR (reverse RGB)

        output = predictor(input_img_bgr)
        car_bbox = output['instances'].pred_boxes.tensor[output['instances'].pred_classes == 2].cpu().numpy()
        car_mask = output['instances'].pred_masks[output['instances'].pred_classes == 2].cpu().numpy()

        save_path = (output_dir / f"{padded_index_string(img_id, index_digits=index_digits)}_inst").with_suffix('.hdf5')
        with h5py.File(str(save_path), "w") as f:
            COMPRESSION = 'lzf'
            f.create_dataset('car_bbox', data=car_bbox, compression=COMPRESSION)
            f.create_dataset('car_mask', data=car_mask, compression=COMPRESSION)
