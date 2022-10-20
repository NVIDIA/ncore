# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import tempfile
import subprocess
import os
import glob 

from PIL import Image
from pathlib import Path

def run_semantic_segmentation(imgs: list, index_digits: int):

    # Create a temporary folder 
    with tempfile.TemporaryDirectory() as temp_dir:
    
        # Save the target resolutions
        img_res = []
        for file in imgs:
            img = Image.open(file)
            w,h = img.size[0], img.size[1]
            img_res.append((w,h))

            # Resize if the image is to large
            if w > 1920 or h > 1280:
                img = img.resize((w//2,h//2), Image.LANCZOS)
            img.save(os.path.join(temp_dir, file.split(os.sep)[-1]), quality=100, subsampling=0)

        args =  f'--dataset cityscapes --cv 0 --fp16 --bs_val 1 --eval folder ' \
                '--eval_folder {} --n_scales 0.5,1.0,2.0 '\
                '--snapshot src/py/deps/semantic_segmentation/pretrained_models/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth '\
                '--arch ocrnet.HRNet_Mscale --result_dir {}'.format(temp_dir, os.path.join(temp_dir,'semantic_seg'))

        # Run the semantic segmentation
        cmd = 'external/semantic-segmentation/train ' + args
        subprocess.Popen(cmd, shell=True).wait()

        predictions = sorted(glob.glob(os.path.join(temp_dir,'semantic_seg','best_images', f"{'?'*index_digits}_prediction.png")))

        assert len(predictions) == len(img_res), "Number of semantic segmentation predictions is not the same as the number of input images"

        for (idx, pred_img), input_image in zip(enumerate(predictions), imgs):
            img = Image.open(pred_img)
            w,h = img.size[0], img.size[1]
            if w != img_res[idx][0] or h != img_res[idx][1]:
                img = img.resize(img_res[idx], Image.LANCZOS)

            input_dir, img_name = os.path.split(input_image)
            img_name = img_name.split('.')[0]
            save_path = Path(os.path.join(input_dir,f"{img_name}_sem")).with_suffix('.png')
            img.save(save_path)