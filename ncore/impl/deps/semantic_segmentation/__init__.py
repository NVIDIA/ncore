# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import tempfile
import subprocess
import os
import glob

from pathlib import Path
from typing import Tuple

from PIL import Image

from ncore.impl.data.types import EncodedImageHandle
from ncore.impl.data.util import padded_index_string


def run_semantic_segmentation(image_handles: list[Tuple[int, EncodedImageHandle]], output_dir: Path, index_digits: int):

    # Create a temporary folder
    with tempfile.TemporaryDirectory() as temp_dir:

        # Save the target resolutions
        img_res = []
        for image_handle in image_handles:
            img_id = image_handle[0]
            img_data = image_handle[1].get_data()
            img_format = img_data.get_encoded_image_format()
            img = img_data.get_decoded_image()

            w, h = img.size[0], img.size[1]
            img_res.append((w, h))

            # Resize if the image is to large
            if w > 1920 or h > 1280:
                img = img.resize((w // 2, h // 2), Image.LANCZOS)
            img.save(
                os.path.join(temp_dir, f"{padded_index_string(img_id, index_digits=index_digits)}.{img_format}"),
                quality=100,
                subsampling=0,
            )

        args = (
            f"--dataset cityscapes --cv 0 --fp16 --bs_val 1 --eval folder "
            "--eval_folder {} --n_scales 0.5,1.0,2.0 "
            "--snapshot external/semantic-segmentation-models/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth "
            "--arch ocrnet.HRNet_Mscale --result_dir {}".format(temp_dir, os.path.join(temp_dir, "semantic_seg"))
        )

        # Run the semantic segmentation
        cmd = "external/semantic-segmentation/train " + args
        subprocess.Popen(cmd, shell=True).wait()

        predictions = sorted(
            glob.glob(os.path.join(temp_dir, "semantic_seg", "best_images", f"{'?'*index_digits}_prediction.png"))
        )

        assert len(predictions) == len(
            img_res
        ), "Number of semantic segmentation predictions is not the same as the number of input images"

        for (idx, pred_img), image_handle in zip(enumerate(predictions), image_handles):
            img = Image.open(pred_img)

            # Resize to original resolution
            w, h = img.size[0], img.size[1]
            if w != img_res[idx][0] or h != img_res[idx][1]:
                img = img.resize(img_res[idx], Image.LANCZOS)

            save_path = (
                output_dir / f"{padded_index_string(image_handle[0], index_digits=index_digits)}_sem"
            ).with_suffix(".png")
            img.save(save_path)
